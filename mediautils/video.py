import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from zoneinfo import ZoneInfo

from moviepy import VideoFileClip

from mediautils.image import set_file_creation_time

_EXIFTOOL_DATE = re.compile(r"^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}")
# Groups whose date tags are not embedded metadata (filesystem times are set
# separately, the others are computed by exiftool).
_NON_EMBEDDED_GROUPS = {"System", "Composite", "ExifTool", "File"}
# Date tags that devices store in UTC without saying so (exiftool shows them
# naive), as opposed to local wall-clock time.
_UTC_TAGS = {"Panasonic:TimeStamp"}


def set_time_video(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a copy of a video with its timestamps set to `dt`.

    Sets the filesystem modification/access times and the MP4 container
    ``creation_time`` metadata (via ffmpeg).

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source video.
    dt : datetime
        The datetime to write into the metadata.
    out_dir : str | os.PathLike
        Directory where the modified copy is saved. Created if it does not exist.
    out_name : str | None
        Output file name. If ``None``, the original file name is kept.

    Returns
    -------
    Path
        Path to the output file.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)

    copy2(path, out_path)

    mod_time = dt.timestamp()
    os.utime(out_path, (mod_time, mod_time))

    metadata_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    tmp_path = out_path.with_stem(out_path.stem + "_tmp")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(out_path),
            "-metadata",
            f"creation_time={metadata_date}",
            "-codec",
            "copy",
            "-y",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    tmp_path.replace(out_path)

    return out_path


def _read_date_tags(path: Path) -> dict[str, str]:
    """Read the embedded date tags of a file with exiftool.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    dict[str, str]
        Maps ``Group:Tag`` names (exiftool family-1 groups, e.g.
        ``Track1:MediaCreateDate``) to their values, QuickTime UTC dates being
        converted to local time with an explicit offset.
    """
    result = subprocess.run(
        [
            "exiftool",
            "-j",
            "-a",
            "-G1",
            "-time:all",
            "-api",
            "QuickTimeUTC=1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    tags = json.loads(result.stdout.decode("utf-8"))[0]
    return {
        name: value
        for name, value in tags.items()
        if ":" in name
        and name.split(":")[0] not in _NON_EMBEDDED_GROUPS
        and isinstance(value, str)
        and _EXIFTOOL_DATE.match(value)
    }


def _parse_exiftool_date(value: str) -> datetime:
    """Parse an exiftool date, with or without UTC offset and fractional seconds.

    Examples
    --------
        >>> _parse_exiftool_date("2019:09:24 16:40:38+02:00")
        datetime.datetime(2019, 9, 24, 16, 40, 38, tzinfo=datetime.timezone(datetime.timedelta(seconds=7200)))

        >>> _parse_exiftool_date("2019:09:24 16:40:38.000")
        datetime.datetime(2019, 9, 24, 16, 40, 38)
    """
    match = re.match(
        r"^(\d{4}):(\d{2}):(\d{2}) (\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:\d{2})?$",
        value,
    )
    if not match:
        raise ValueError(f"Unexpected exiftool date: {value}")
    year, month, day, hms, offset = match.groups()
    offset = "+00:00" if offset == "Z" else (offset or "")
    return datetime.fromisoformat(f"{year}-{month}-{day}T{hms}{offset}")


def _wall_clock(name: str, value: str, file_offset: str | None) -> datetime | None:
    """Return the wall-clock datetime a date tag denotes, without timezone.

    For a tag carrying an offset, the offset is dropped. For a tag stored in
    UTC by the device (see `_UTC_TAGS`), the value is converted with the
    offset the device recorded in the file; ``None`` if there is none.

    Examples
    --------
        >>> _wall_clock("XML:ShootStartDate", "2020:04:20 16:27:48+01:00", None)
        datetime.datetime(2020, 4, 20, 16, 27, 48)

        >>> _wall_clock("Panasonic:TimeStamp", "2020:04:20 15:27:48", "+01:00")
        datetime.datetime(2020, 4, 20, 16, 27, 48)

        >>> _wall_clock("Panasonic:TimeStamp", "2020:04:20 15:27:48", None) is None
        True
    """
    parsed = _parse_exiftool_date(value)
    if name in _UTC_TAGS:
        if file_offset is None:
            return None
        offset = datetime.strptime(file_offset, "%z").tzinfo
        parsed = parsed.replace(tzinfo=timezone.utc).astimezone(offset)
    return parsed.replace(tzinfo=None)


def _read_file_offset(path: Path) -> str | None:
    """Read the UTC offset a camera recorded in a file (``OffsetTime``), if any."""
    result = subprocess.run(
        ["exiftool", "-s3", "-OffsetTimeOriginal", "-OffsetTime", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.split()
    return lines[0] if lines else None


def _set_all_dates_video(
    path: Path, dt: datetime, out_path: Path, tz: str
) -> dict[str, str]:
    """Do the work of :func:`set_all_dates_video`.

    Returns
    -------
    dict[str, str]
        The date tags exiftool cannot write but that already hold the
        wall-clock time `dt`, with their values.
    """
    copy2(path, out_path)
    aware = dt.replace(tzinfo=ZoneInfo(tz))
    value = aware.isoformat(sep=" ").replace("-", ":", 2)  # 2019:09:24 16:40:38+02:00
    before = _read_date_tags(out_path)
    if before:
        subprocess.run(
            ["exiftool", "-overwrite_original", "-api", "QuickTimeUTC=1"]
            + [f"-{name}={value}" for name in before]
            + [str(out_path)],
            check=True,
            capture_output=True,
        )
    file_offset = None
    unwritable, wrong = {}, {}
    for name, read in _read_date_tags(out_path).items():
        parsed = _parse_exiftool_date(read)
        if parsed == (aware if parsed.tzinfo else dt):
            continue
        if read != before.get(name):
            wrong[name] = read  # written, but with a wrong value: a bug
            continue
        if name in _UTC_TAGS and file_offset is None:
            file_offset = _read_file_offset(out_path)
        if _wall_clock(name, read, file_offset) == dt:
            unwritable[name] = read
        else:
            wrong[name] = read
    if wrong:
        raise RuntimeError(f"wrong dates left: {wrong}")

    mod_time = dt.timestamp()
    os.utime(out_path, (mod_time, mod_time))
    if sys.platform == "win32":
        set_file_creation_time(out_path, dt)
    return unwritable


def set_all_dates_video(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
    tz: str = "Europe/Paris",
) -> tuple[Path, dict[str, str]]:
    """Write a copy of a video with every date it carries set to `dt`.

    Every date tag already present in the file (container and track dates,
    Keys/ItemList/XMP dates, ...) is overwritten with `dt`; no tag is created,
    since an MP4/MOV always carries its container and track dates. The
    filesystem modification, access and (on Windows) creation times are set
    too. The tags are read back afterwards:

    - a tag holding `dt` is fine;
    - a tag exiftool cannot write (e.g. Panasonic's ``TimeStamp`` or the dates
      of the XML clip metadata of Lumix cameras) is accepted if it already
      denotes the wall-clock time `dt` (offset ignored; a tag stored in UTC by
      the camera is converted with the offset the camera recorded), and
      reported;
    - any other tag makes the whole operation fail, and no copy is left.

    QuickTime dates are stored in UTC, as phones do; `dt` is converted using
    the UTC offset of `tz` at that date (daylight saving time included).
    Requires ``exiftool`` on `PATH`. Unlike :func:`set_time_video`, the video
    is not remuxed.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source video.
    dt : datetime
        The datetime to write, naive, as local wall-clock time in `tz`.
    out_dir : str | os.PathLike
        Directory where the modified copy is saved. Created if it does not exist.
    out_name : str | None
        Output file name. If ``None``, the original file name is kept.
    tz : str
        IANA timezone in which `dt` is expressed.

    Returns
    -------
    tuple[Path, dict[str, str]]
        Path to the output file, and the tags exiftool could not write but
        that already denote `dt` (name to value).

    Raises
    ------
    RuntimeError
        If a date tag does not denote `dt` after writing (the copy is removed).
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)
    try:
        unwritable = _set_all_dates_video(path, dt, out_path, tz)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name}: {error}") from error
    return out_path, unwritable


def rotate_video(
    path: str | os.PathLike,
    angle: int = 90,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a rotated copy of a video.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source video.
    angle : int
        Rotation angle in degrees (90, 180, or 270).
    out_dir : str | os.PathLike
        Directory where the rotated video is saved. Created if it does not exist.
    out_name : str | None
        Output file name. If ``None``, the original file name is kept.

    Returns
    -------
    Path
        Path to the output file.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)

    clip = VideoFileClip(str(path))
    rotated_clip = clip.rotated(angle)
    rotated_clip.write_videofile(str(out_path), codec="libx264")
    rotated_clip.close()
    clip.close()

    return out_path
