import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import copy2
from zoneinfo import ZoneInfo

from moviepy import VideoFileClip

from mediautils.image import set_file_creation_time

_EXIFTOOL_DATE = re.compile(r"^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}")
# Groups whose date tags are not embedded metadata (filesystem times are set
# separately, the others are computed by exiftool).
_NON_EMBEDDED_GROUPS = {"System", "Composite", "ExifTool", "File"}


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


def set_all_dates_video(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
    tz: str = "Europe/Paris",
) -> Path:
    """Write a copy of a video with every date it carries set to `dt`.

    Every date tag already present in the file (container and track dates,
    Keys/ItemList/XMP dates, ...) is overwritten with `dt`; no tag is created,
    since an MP4/MOV always carries its container and track dates. The
    filesystem modification, access and (on Windows) creation times are set
    too. The tags are read back afterwards, and an error is raised if any of
    them does not hold `dt`.

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
    Path
        Path to the output file.

    Raises
    ------
    RuntimeError
        If a date tag does not hold `dt` after writing.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)
    copy2(path, out_path)

    aware = dt.replace(tzinfo=ZoneInfo(tz))
    value = aware.isoformat(sep=" ").replace("-", ":", 2)  # 2019:09:24 16:40:38+02:00
    tags = _read_date_tags(out_path)
    if tags:
        subprocess.run(
            ["exiftool", "-overwrite_original", "-api", "QuickTimeUTC=1"]
            + [f"-{name}={value}" for name in tags]
            + [str(out_path)],
            check=True,
            capture_output=True,
        )
    wrong = {}
    for name, read in _read_date_tags(out_path).items():
        parsed = _parse_exiftool_date(read)
        expected = aware if parsed.tzinfo else dt
        if parsed != expected:
            wrong[name] = read
    if wrong:
        raise RuntimeError(f"Dates not set as expected in {out_path.name}: {wrong}")

    mod_time = dt.timestamp()
    os.utime(out_path, (mod_time, mod_time))
    if sys.platform == "win32":
        set_file_creation_time(out_path, dt)

    return out_path


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
