import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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
# Accepted gap between an unwritable date tag and the target datetime: a Lumix
# truncates sub-seconds in its EXIF dates but rounds them in its XML dates.
_UNWRITABLE_TOLERANCE = timedelta(seconds=1)
# Tags of the XMP "Pantry" of Adobe exports (Premiere...), which holds a copy of
# the metadata of the source files of a montage; removed as a whole.
_PANTRY_PREFIX = "XMP-xmpMM:Pantry"
# Lists of dates kept as they are, as an explicit exception to "every date of
# the file is the date of its name": the save history of Adobe software gives
# the true dates of the editing work, and nothing uses it to date the video.
_KEPT_DATE_LISTS = {"XMP-xmpMM:HistoryWhen"}


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
        converted to local time with an explicit offset. Only single dates:
        see :func:`_read_date_list_tags` for lists of dates.
    """
    return {
        name: value
        for name, value in _read_time_tags(path).items()
        if isinstance(value, str)
        and _EXIFTOOL_DATE.match(value)
        and name not in _KEPT_DATE_LISTS  # a history with a single entry
    }


def _read_date_list_tags(path: Path) -> dict[str, list]:
    """Read the embedded tags of a file holding lists of dates.

    Such tags are typically fields of lists of structures, e.g.
    ``XMP-xmpMM:HistoryWhen`` (the save history of Adobe software).
    """
    return {
        name: value
        for name, value in _read_time_tags(path).items()
        if isinstance(value, list)
        and value
        and all(isinstance(v, str) and _EXIFTOOL_DATE.match(v) for v in value)
    }


def _read_time_tags(path: Path) -> dict:
    """Read the embedded tags of the exiftool "time" group of a file."""
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
        if ":" in name and name.split(":")[0] not in _NON_EMBEDDED_GROUPS
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


def _shift_date_text(text: str, delta: timedelta) -> str:
    """Shift a date written as text, keeping its exact layout and any offset.

    Examples
    --------
        >>> _shift_date_text("2026:04:12 17:07:44", timedelta(minutes=-15))
        '2026:04:12 16:52:44'

        >>> _shift_date_text("2026-06-12T23:40:07+01:00", timedelta(hours=6, minutes=45))
        '2026-06-13T06:25:07+01:00'
    """
    match = re.match(
        r"^(\d{4})([:-])(\d{2})[:-](\d{2})([ T])(\d{2}):(\d{2}):(\d{2})", text
    )
    if not match:
        raise ValueError(f"Unexpected date text: {text}")
    year, sep, month, day, middle, hour, minute, second = match.groups()
    new = datetime(*map(int, (year, month, day, hour, minute, second))) + delta
    return new.strftime(f"%Y{sep}%m{sep}%d{middle}%H:%M:%S") + text[match.end() :]


def _patch_locked_dates(path: Path, tags: dict[str, str], delta: timedelta) -> None:
    """Shift, in place, date tags that exiftool cannot write.

    Such dates (Lumix EXIF block, ``Panasonic:TimeStamp``, XML clip metadata)
    are plain fixed-length text: each is replaced by the shifted text, of the
    same length, so that nothing else in the file moves. Only the top-level
    atoms other than ``mdat`` (the audio/video data) are searched, and each
    date text must be found exactly as many times as there are tags holding
    it, otherwise nothing is changed.

    Parameters
    ----------
    path : Path
        Path to the MP4/MOV file, modified in place.
    tags : dict[str, str]
        The tags to shift, as read by exiftool (name to value).
    delta : timedelta
        The shift to apply.

    Raises
    ------
    RuntimeError
        If a date text is not found the expected number of times.
    """
    expected = {}
    for value in tags.values():
        # exiftool shows XML dates as "YYYY:MM:DD HH:MM:SS+HH:MM", stored
        # "YYYY-MM-DDTHH:MM:SS+HH:MM"; EXIF-like dates are stored as shown.
        if re.search(r"[+-]\d{2}:\d{2}$", value):
            text = value[:4] + "-" + value[5:7] + "-" + value[8:10] + "T" + value[11:]
        else:
            text = value
        expected[text] = expected.get(text, 0) + 1
    with open(path, "r+b") as f:
        atoms = []
        size = os.fstat(f.fileno()).st_size
        pos = 0
        while pos < size:
            f.seek(pos)
            atom_size, kind = struct.unpack(">I4s", f.read(8))
            if atom_size == 1:
                atom_size = struct.unpack(">Q", f.read(8))[0]
            elif atom_size == 0:
                atom_size = size - pos
            if kind != b"mdat":
                f.seek(pos)
                atoms.append([pos, bytearray(f.read(atom_size))])
            pos += atom_size
        for text, count in expected.items():
            old = text.encode("ascii")
            found = sum(content.count(old) for _, content in atoms)
            if found != count:
                raise RuntimeError(
                    f"locked date {text!r} found {found} time(s), expected {count}"
                )
        for text in expected:
            old = text.encode("ascii")
            new = _shift_date_text(text, delta).encode("ascii")
            for atom in atoms:
                atom[1] = atom[1].replace(old, new)
        for start, content in atoms:
            f.seek(start)
            f.write(content)


def _holds(value: str, dt: datetime, aware: datetime) -> bool:
    """Whether an exiftool date already denotes `dt` (to the second).

    Examples
    --------
        >>> dt = datetime(2025, 4, 5, 20, 55, 22)
        >>> aware = dt.replace(tzinfo=ZoneInfo("Europe/Paris"))
        >>> _holds("2025:04:05 20:55:22.668+02:00", dt, aware)
        True
        >>> _holds("0000:00:00 00:00:00", dt, aware)
        False
    """
    try:
        parsed = _parse_exiftool_date(value)
    except ValueError:  # e.g. the zero dates of WhatsApp videos
        return False
    return parsed == (aware if parsed.tzinfo else dt)


def _set_all_dates_video(
    path: Path,
    dt: datetime,
    out_path: Path,
    tz: str,
    locked_shift: timedelta,
    copy: bool = True,
) -> dict[str, str]:
    """Do the work of :func:`set_all_dates_video`.

    With ``copy=False``, `out_path` is modified in place (`path` is ignored).

    Returns
    -------
    dict[str, str]
        The date tags exiftool cannot write, with their final values.
    """
    if copy:
        copy2(path, out_path)
    aware = dt.replace(tzinfo=ZoneInfo(tz))
    value = aware.isoformat(sep=" ").replace("-", ":", 2)  # 2019:09:24 16:40:38+02:00
    before = _read_date_tags(out_path)
    if any(name.startswith(_PANTRY_PREFIX) for name in before):
        # Adobe's copy of the metadata of the source files of a montage: their
        # dates are not those of this video, and exiftool cannot write them.
        subprocess.run(
            ["exiftool", "-overwrite_original", "-XMP-xmpMM:Pantry=", str(out_path)],
            check=True,
            capture_output=True,
        )
        before = _read_date_tags(out_path)
    lists = {
        name: value
        for name, value in _read_date_list_tags(out_path).items()
        if name not in _KEPT_DATE_LISTS
    }
    if lists:
        raise RuntimeError(f"lists of dates not handled: {lists}")
    # Tags already holding `dt` are left alone, which keeps any fraction of a
    # second they carry (see :func:`write_precise_start`).
    to_write = [name for name, read in before.items() if not _holds(read, dt, aware)]
    if to_write:
        subprocess.run(
            ["exiftool", "-overwrite_original", "-api", "QuickTimeUTC=1"]
            + [f"-{name}={value}" for name in to_write]
            + [str(out_path)],
            check=True,
            capture_output=True,
        )
    unwritable, wrong = {}, {}
    for name, read in _read_date_tags(out_path).items():
        parsed = _parse_exiftool_date(read)
        if parsed == (aware if parsed.tzinfo else dt):
            continue
        if read != before.get(name):
            wrong[name] = read  # written, but with a wrong value: a bug
        else:
            unwritable[name] = read
    if wrong:
        raise RuntimeError(f"wrong dates left: {wrong}")
    if unwritable and locked_shift:
        _patch_locked_dates(out_path, unwritable, locked_shift)
        after = _read_date_tags(out_path)
        unwritable = {name: after.get(name, "") for name in unwritable}
    file_offset = _read_file_offset(out_path) if unwritable else None
    for name, read in unwritable.items():
        wall = _wall_clock(name, read, file_offset) if read else None
        # Devices truncate or round sub-seconds depending on the tag.
        if wall is None or abs(wall - dt) > _UNWRITABLE_TOLERANCE:
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
    locked_shift: timedelta = timedelta(0),
) -> tuple[Path, dict[str, str]]:
    """Write a copy of a video with every date it carries set to `dt`.

    Every date tag already present in the file (container and track dates,
    Keys/ItemList/XMP dates, ...) is overwritten with `dt`; no tag is created,
    since an MP4/MOV always carries its container and track dates. The XMP
    "Pantry" of Adobe exports, a copy of the metadata of the source files of a
    montage (whose dates are not those of the video), is removed. Lists of
    dates are left as they are if known to be harmless (the save history of
    Adobe software, which gives the true dates of the editing work), and make
    the operation fail otherwise. The
    filesystem modification, access and (on Windows) creation times are set
    too. The tags are read back afterwards:

    - a tag holding `dt` is fine;
    - a tag exiftool cannot write (e.g. the EXIF block, Panasonic's
      ``TimeStamp`` or the XML clip metadata of Lumix cameras) is first
      shifted by `locked_shift`, if any, directly in the file (see
      :func:`_patch_locked_dates`); it is then accepted if it denotes the
      wall-clock time `dt`, give or take one second (offset ignored; a tag
      stored in UTC by the camera is converted with the offset the camera
      recorded), and reported;
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
    locked_shift : timedelta
        Shift to apply to the date tags exiftool cannot write: the difference
        between `dt` and the datetime these tags hold (e.g. the correction of
        a camera clock that was off).

    Returns
    -------
    tuple[Path, dict[str, str]]
        Path to the output file, and the tags exiftool could not write, with
        their final values (name to value).

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
        unwritable = _set_all_dates_video(path, dt, out_path, tz, locked_shift)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name}: {error}") from error
    return out_path, unwritable


def _timecode(time_of_day: float, frame_rate: float) -> tuple[str, str] | None:
    """Time-of-day timecode at `frame_rate`, in the XMP (xmpDM) conventions.

    29.97 and 59.94 fps use drop-frame timecode, which stays aligned with the
    wall clock. Returns ``None`` for other non-integer rates.

    Examples
    --------
        >>> _timecode(20 * 3600 + 55 * 60 + 22.668, 25)
        ('25Timecode', '20:55:22:17')

        Drop-frame labels drift ahead of the wall clock by about 3.6 ms per
        hour (here 75 ms, 4.5 frames), but they convert back exactly to the
        frame count, hence to the time:

        >>> _timecode(20 * 3600 + 55 * 60 + 22.668, 60000 / 1001)
        ('5994DropTimecode', '20:55:22;45')

        Frame 1800 at 29.97 fps is the first of minute 1, whose labels ;00 and
        ;01 are dropped:

        >>> _timecode(1799 * 1.001 / 30, 30000 / 1001), _timecode(1800 * 1.001 / 30, 30000 / 1001)
        (('2997DropTimecode', '00:00:59;29'), ('2997DropTimecode', '00:01:00;02'))

        >>> _timecode(3600.0, 33.2) is None
        True
    """
    nominal = round(frame_rate)
    if abs(frame_rate - nominal) < 1e-6:
        frames = round(time_of_day * nominal)
        seconds, ff = divmod(frames, nominal)
        h, rest = divmod(seconds, 3600)
        m, s = divmod(rest, 60)
        return f"{nominal}Timecode", f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"
    if nominal in (30, 60) and abs(frame_rate - nominal * 1000 / 1001) < 1e-3:
        # Drop-frame: labels 0 and 1 (2 and 3 at 59.94) are skipped every
        # minute, except every tenth minute.
        drop = 2 * nominal // 30
        frames = round(time_of_day * frame_rate)
        per_10min = 10 * 60 * nominal - 9 * drop
        per_min = 60 * nominal - drop
        tens, rest = divmod(frames, per_10min)
        frames += 9 * drop * tens
        if rest > drop:
            frames += drop * ((rest - drop) // per_min)
        ff = frames % nominal
        s = frames // nominal % 60
        m = frames // (nominal * 60) % 60
        h = frames // (nominal * 3600)
        time_format = "2997DropTimecode" if nominal == 30 else "5994DropTimecode"
        return time_format, f"{h:02d}:{m:02d}:{s:02d};{ff:02d}"
    return None


def write_precise_start(
    path: str | os.PathLike,
    start: datetime,
    frame_rate: float | None = None,
    tz: str = "Europe/Paris",
    precise: bool = True,
) -> None:
    """Record, in place, the start of a video more precisely than to the second.

    Writes in the XMP metadata of the file (``xmpDM:shotDate``) the start of
    the recording to the millisecond and, if `frame_rate` is given and
    supported, a time-of-day start timecode to the frame
    (``xmpDM:startTimecode``), which editing software can use to synchronize
    several cameras. Nothing else in the file changes (no remuxing). The name
    and the other dates of the file, which stay to the second, are left alone.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the video, modified in place.
    start : datetime
        Start of the recording, naive, as local wall-clock time in `tz`.
    frame_rate : float | None
        Frame rate of the video, for the timecode (none if ``None``).
    tz : str
        IANA timezone in which `start` is expressed.
    precise : bool
        If ``False`` (start only known to about a second), the date is written
        to the second and no timecode is written.
    """
    aware = start.replace(tzinfo=ZoneInfo(tz))
    stamp = aware.strftime("%Y:%m:%d %H:%M:%S")
    if precise:
        stamp += f".{start.microsecond // 1000:03d}"
    offset = aware.strftime("%z")
    args = [f"-XMP-xmpDM:ShotDate={stamp}{offset[:3]}:{offset[3:]}"]
    if precise and frame_rate:
        midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
        tc = _timecode((start - midnight).total_seconds(), frame_rate)
        if tc:
            args += [
                # "#=": raw XMP value, not exiftool's printed one ("25 fps").
                f"-XMP-xmpDM:StartTimecodeTimeFormat#={tc[0]}",
                f"-XMP-xmpDM:StartTimecodeTimeValue={tc[1]}",
            ]
    subprocess.run(
        ["exiftool", "-overwrite_original", *args, str(path)],
        check=True,
        capture_output=True,
    )


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
