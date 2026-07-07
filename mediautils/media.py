import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from shutil import copy2
from zoneinfo import ZoneInfo

from mediautils.file_name import (
    _date_prefix_to_date,
    file_name_to_datetime,
    replace_datetime_prefix,
    wa_file_name_to_date,
)
from mediautils.image import (
    get_offset_image,
    get_time_image,
    set_file_creation_time,
    set_time_image,
    set_time_raw,
)
from mediautils.video import set_time_video

_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
_RAW_EXTENSIONS = {".rw2"}
_COPY_ONLY_EXTENSIONS = {".png"}
_VIDEO_EXTENSIONS = {".mp4", ".mov"}
_XMP_EXTENSIONS = {".xmp"}


def _end_of_day_time(index: int, count: int) -> time:
    """Return a time near end of day for the given index among `count` items.

    Timestamps are placed at the end of the day, counting backwards from
    23:59:59 so that all `count` items fit with one-second spacing.

    Parameters
    ----------
    index : int
        0-based index of the item.
    count : int
        Total number of items to fit.

    Returns
    -------
    time

    Raises
    ------
    ValueError
        If `count` exceeds 86400 (the number of seconds in a day).

    Examples
    --------
        >>> _end_of_day_time(0, 3)
        datetime.time(23, 59, 57)

        >>> _end_of_day_time(2, 3)
        datetime.time(23, 59, 59)

        >>> _end_of_day_time(0, 1)
        datetime.time(23, 59, 59)

        >>> _end_of_day_time(0, 61)
        datetime.time(23, 58, 59)
    """
    if count > 86400:
        raise ValueError(f"Too many files for a single day: {count}")
    start_second = 86400 - count
    total_seconds = start_second + index
    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    second = total_seconds % 60
    return time(hour=hour, minute=minute, second=second)


def set_time(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a copy of an image or video with its timestamps set to `dt`.

    Dispatches to :func:`~mediautils.image.set_time_image` or
    :func:`~mediautils.video.set_time_video` based on the file extension.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source file.
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

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        return set_time_image(path, dt, out_dir=out_dir, out_name=out_name)
    if ext in _RAW_EXTENSIONS:
        return set_time_raw(path, dt, out_dir=out_dir, out_name=out_name)
    if ext in _VIDEO_EXTENSIONS:
        return set_time_video(path, dt, out_dir=out_dir, out_name=out_name)
    if ext in _COPY_ONLY_EXTENSIONS:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (out_name or path.name)
        copy2(path, out_path)
        mod_time = dt.timestamp()
        os.utime(out_path, (mod_time, mod_time))
        return out_path
    raise ValueError(f"Unsupported file extension: {ext}")


def process_datetime_filenames(
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
) -> list[Path]:
    """Process files with standardized names: set timestamps from file names.

    For each file in `in_dir` whose name starts with ``YYYYMMDD_HHMMSS``,
    writes a copy to `out_dir` with metadata matching the datetime encoded
    in the file name.

    Parameters
    ----------
    in_dir : str | os.PathLike
        Directory containing media files with standardized names.
    out_dir : str | os.PathLike
        Directory where processed files are saved.

    Returns
    -------
    list[Path]
        Paths to the output files.
    """
    in_dir = Path(in_dir)
    results = []
    for path in sorted(in_dir.iterdir()):
        if not path.is_file():
            continue
        print(path.name)
        dt = file_name_to_datetime(path)
        result = set_time(path, dt, out_dir=out_dir)
        results.append(result)
    print("Done!")
    return results


def process_whatsapp_files(
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
) -> list[Path]:
    """Process WhatsApp media files: set timestamps and rename.

    For each file in `in_dir`, extracts the date from the WhatsApp file name,
    assigns a time near end of day (incrementing by one second per file
    within the same date), renames to ``YYYYMMDD_HHMMSS_originalname``, and
    writes the result to `out_dir` with updated metadata.

    Parameters
    ----------
    in_dir : str | os.PathLike
        Directory containing WhatsApp media files.
    out_dir : str | os.PathLike
        Directory where processed files are saved.

    Returns
    -------
    list[Path]
        Paths to the output files.
    """
    in_dir = Path(in_dir)
    entries = [(p, wa_file_name_to_date(p)) for p in in_dir.iterdir() if p.is_file()]
    entries.sort(key=lambda e: (e[1], e[0].name))

    groups: dict[object, list[Path]] = {}
    for path, d in entries:
        groups.setdefault(d, []).append(path)

    results = []
    for d, paths in groups.items():
        for i, path in enumerate(paths):
            print(path.name)
            t = _end_of_day_time(i, len(paths))
            dt = datetime.combine(d, t)
            out_name = f"{dt.strftime('%Y%m%d_%H%M%S')}_{path.name}"
            result = set_time(path, dt, out_dir=out_dir, out_name=out_name)
            results.append(result)

    print("Done!")
    return results


def gather_yes_no(
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
) -> list[Path]:
    """Copy culled files from nested ``yes``/``no`` subdirectories to output buckets.

    Recursively scans `in_dir` for files whose immediate parent directory is
    named ``yes`` or ``no``, and copies them to ``out_dir/yes`` or
    ``out_dir/no`` respectively. Files in any other directory (e.g. ``temp``)
    are ignored.

    Parameters
    ----------
    in_dir : str | os.PathLike
        Root directory to scan recursively.
    out_dir : str | os.PathLike
        Output root. ``yes`` and ``no`` subdirectories are created as needed.

    Returns
    -------
    list[Path]
        Paths to the copied output files.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    results = []
    skipped = 0
    for path in sorted(in_dir.rglob("*")):
        if not path.is_file():
            continue
        bucket = path.parent.name
        if bucket not in {"yes", "no"}:
            skipped += 1
            continue
        print(path.name)
        dest_dir = out_dir / bucket
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        copy2(path, dest)
        results.append(dest)
    if skipped:
        print(f"Warning: {skipped} file(s) ignored (not in a 'yes' or 'no' directory).")
    else:
        print("All files were in a 'yes' or 'no' directory.")
    print("Done!")
    return results


def process_date_subdirs(
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
) -> list[Path]:
    """Process files organized in date-named subdirectories.

    Each subdirectory of `in_dir` must have a name starting with ``YYYYMMDD``.
    Files inside each subdirectory are assigned timestamps near end of day
    (incrementing by one second per file), renamed to
    ``YYYYMMDD_HHMMSS_originalname``, and written to `out_dir` with updated
    metadata.

    Parameters
    ----------
    in_dir : str | os.PathLike
        Directory containing date-named subdirectories.
    out_dir : str | os.PathLike
        Directory where processed files are saved.

    Returns
    -------
    list[Path]
        Paths to the output files.
    """
    in_dir = Path(in_dir)
    results = []

    for sub_dir in sorted(in_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        d = _date_prefix_to_date(sub_dir.name)
        files = sorted(p for p in sub_dir.iterdir() if p.is_file())
        for i, path in enumerate(files):
            print(path.name)
            t = _end_of_day_time(i, len(files))
            dt = datetime.combine(d, t)
            out_name = f"{dt.strftime('%Y%m%d_%H%M%S')}_{path.name}"
            result = set_time(path, dt, out_dir=out_dir, out_name=out_name)
            results.append(result)

    print("Done!")
    return results


def _offset_str_to_timedelta(offset: str) -> timedelta:
    """Parse an EXIF UTC offset string into a timedelta.

    Parameters
    ----------
    offset : str
        UTC offset in ``"+HH:MM"``/``"-HH:MM"`` format (e.g. ``"+09:00"``).

    Returns
    -------
    timedelta

    Examples
    --------
        >>> _offset_str_to_timedelta("+09:00")
        datetime.timedelta(seconds=32400)

        >>> _offset_str_to_timedelta("-05:30")
        datetime.timedelta(days=-1, seconds=66600)
    """
    sign = -1 if offset[0] == "-" else 1
    hours, minutes = offset[1:].split(":")
    return sign * timedelta(hours=int(hours), minutes=int(minutes))


def _timedelta_to_offset_str(delta: timedelta) -> str:
    """Format a timedelta as an EXIF UTC offset string.

    Inverse of :func:`_offset_str_to_timedelta`.

    Parameters
    ----------
    delta : timedelta
        UTC offset.

    Returns
    -------
    str
        UTC offset in ``"+HH:MM"``/``"-HH:MM"`` format (e.g. ``"+09:00"``).

    Examples
    --------
        >>> _timedelta_to_offset_str(timedelta(hours=9))
        '+09:00'

        >>> _timedelta_to_offset_str(timedelta(hours=-5, minutes=-30))
        '-05:30'
    """
    total_minutes = round(delta.total_seconds() / 60)
    sign = "-" if total_minutes < 0 else "+"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def process_camera_timezone(
    camera_timezone: str,
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
    real_timezone: str = "Europe/Paris",
) -> list[Path]:
    """Correct photos from a camera left on a foreign time zone.

    For a camera whose clock was never switched from `camera_timezone`
    while actually being used in `real_timezone`, corrects every file in
    `in_dir` with a standardized name (``YYYYMMDD_HHMMSS...``): the
    datetime encoded in the file name's prefix is reinterpreted as a
    `camera_timezone` wall-clock reading, then converted — via the IANA
    time zone database, so daylight-saving time is resolved per the
    photo's own date — to the corresponding `real_timezone` wall-clock time
    and UTC offset.

    Unlike :func:`process_timezone_shift`, no reference photo is needed:
    both time zones are known upfront, so the correction is computed
    directly and stays correct even if `in_dir` spans a DST transition in
    either zone (the US and the EU do not switch to/from DST on the same
    dates, so the gap between them is not always the same number of hours).

    For each file, writes a copy to `out_dir` with:

    - the file name's ``YYYYMMDD_HHMMSS`` prefix replaced by the corrected
      datetime — anything after the prefix (e.g. the camera's own original
      file name, kept as an identifier back to the source device) is left
      untouched, see :func:`~mediautils.file_name.replace_datetime_prefix`;
    - for JPEG/RAW images, ``datetime_original``/``DateTimeOriginal`` and
      ``offset_time_original``/``OffsetTimeOriginal`` set to the corrected
      `real_timezone` datetime and UTC offset; videos have their
      ``creation_time`` set via :func:`set_time` (no offset tag);
    - the copy's filesystem creation, modification and access times set to
      match, for the same reasons as :func:`process_clock_shift`.

    ``.xmp`` sidecars are not supported: the text-substitution approach used
    by :func:`process_clock_shift` assumes the offset embedded in the file
    does not change, which is false for a genuine time zone correction.

    Parameters
    ----------
    camera_timezone : str
        IANA time zone the camera's clock was left on (e.g.
        ``"America/New_York"``).
    in_dir : str | os.PathLike
        Directory containing the files to correct, with standardized names.
    out_dir : str | os.PathLike
        Directory where corrected copies are saved.
    real_timezone : str
        IANA time zone the photos were actually taken in.

    Returns
    -------
    list[Path]
        Paths to the output files.

    Raises
    ------
    ValueError
        If a file has an unsupported extension, including ``.xmp``.
    """
    in_dir = Path(in_dir)
    results = []
    for path in sorted(p for p in in_dir.iterdir() if p.is_file()):
        print(path.name)
        ext = path.suffix.lower()
        if ext in _XMP_EXTENSIONS:
            raise ValueError(
                f"{path}: .xmp sidecars are not supported by process_camera_timezone"
            )

        old_dt = file_name_to_datetime(path)
        wrong_aware = old_dt.replace(tzinfo=ZoneInfo(camera_timezone))
        new_aware = wrong_aware.astimezone(ZoneInfo(real_timezone))
        new_dt = new_aware.replace(tzinfo=None)
        offset = _timedelta_to_offset_str(new_aware.utcoffset())
        out_name = replace_datetime_prefix(path, new_dt)

        if ext in _IMAGE_EXTENSIONS:
            result = set_time_image(
                path, new_dt, out_dir=out_dir, out_name=out_name, offset=offset
            )
        elif ext in _RAW_EXTENSIONS:
            result = set_time_raw(
                path, new_dt, out_dir=out_dir, out_name=out_name, offset=offset
            )
        else:
            result = set_time(path, new_dt, out_dir=out_dir, out_name=out_name)

        set_file_creation_time(result, new_dt)
        mod_time = new_dt.timestamp()
        os.utime(result, (mod_time, mod_time))
        results.append(result)
    print("Done!")
    return results


def process_timezone_shift(
    phone_reference: str | os.PathLike,
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
    home_timezone: str = "Europe/Paris",
) -> list[Path]:
    """Correct the recorded time of images from a camera left on the wrong timezone.

    For a trip where the camera's clock was never updated from `home_timezone`
    while the phone auto-updated to the destination's local time, `phone_reference`
    (any photo taken on the phone during the trip, with correct local time and
    an ``offset_time_original`` EXIF tag) is enough to derive both the
    destination's UTC offset and the camera's clock error: the true instant of
    `phone_reference` is computed from its own datetime and offset, then
    compared to what a `home_timezone` clock would have displayed at that same
    instant. That gap, plus the destination offset, are applied to every image
    in `in_dir`: for each one, reads its ``datetime_original``, shifts it by
    the gap, and writes a copy to `out_dir` with the corrected datetime and
    the destination ``offset_time_original``. The copy's filesystem creation,
    modification and access times are also set to match (see
    :func:`~mediautils.image.set_file_creation_time`), since tools like
    FastStone Image Viewer's rename feature read those instead of the EXIF
    metadata. File names are left untouched.

    Parameters
    ----------
    phone_reference : str | os.PathLike
        Path to any phone photo taken during the trip, with correct local
        time and an ``offset_time_original`` EXIF tag.
    in_dir : str | os.PathLike
        Directory containing the camera images to correct.
    out_dir : str | os.PathLike
        Directory where corrected copies are saved.
    home_timezone : str
        IANA timezone the camera's clock was left on (e.g. ``"Europe/Paris"``).

    Returns
    -------
    list[Path]
        Paths to the output files.

    Raises
    ------
    ValueError
        If `in_dir` contains no files, or if `phone_reference` has no
        ``offset_time_original`` EXIF tag.
    NotImplementedError
        If not running on Windows.
    """
    in_dir = Path(in_dir)
    files = sorted(p for p in in_dir.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"No files found in {in_dir}")

    offset = get_offset_image(phone_reference)
    if offset is None:
        raise ValueError(f"{phone_reference} has no offset_time_original EXIF tag")

    reference_utc = get_time_image(phone_reference) - _offset_str_to_timedelta(offset)
    home_offset = (
        reference_utc.replace(tzinfo=timezone.utc)
        .astimezone(ZoneInfo(home_timezone))
        .utcoffset()
    )
    delta = _offset_str_to_timedelta(offset) - home_offset

    results = []
    for path in files:
        print(path.name)
        dt = get_time_image(path) + delta
        result = set_time_image(path, dt, out_dir=out_dir, offset=offset)
        set_file_creation_time(result, dt)
        mod_time = dt.timestamp()
        os.utime(result, (mod_time, mod_time))
        results.append(result)
    print("Done!")
    return results


def _shift_xmp_datetimes(
    path: str | os.PathLike,
    old_dt: datetime,
    new_dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a copy of an XMP sidecar file with its embedded dates shifted.

    XMP sidecars (e.g. Lightroom edits) are plain XML text repeating the
    photo's datetime as an ISO 8601 string (e.g. in ``xmp:ModifyDate``,
    ``xmp:CreateDate``, ``exif:DateTimeOriginal``, ``photoshop:DateCreated``)
    and, in ``crs:RawFileName``, the standardized ``YYYYMMDD_HHMMSS`` prefix
    of the associated RAW file's name. Rather than parsing the XML, this
    performs a plain text substitution of both formats: every occurrence of
    `old_dt`'s encoding is replaced by `new_dt`'s. Any sub-second precision
    or UTC offset trailing the ISO timestamp is left untouched, since only
    the matched prefix is replaced.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source .xmp file.
    old_dt : datetime
        The datetime currently encoded in the file (matches its file name).
    new_dt : datetime
        The corrected datetime to substitute in.
    out_dir : str | os.PathLike
        Directory where the modified copy is saved. Created if it does not exist.
    out_name : str | None
        Output file name. If ``None``, the original file name is kept.

    Returns
    -------
    Path
        Path to the output file.

    Raises
    ------
    ValueError
        If the ISO 8601 encoding of `old_dt` is not found in the file's content.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)

    text = path.read_text(encoding="utf-8-sig")
    old_iso = old_dt.strftime("%Y-%m-%dT%H:%M:%S")
    new_iso = new_dt.strftime("%Y-%m-%dT%H:%M:%S")
    if old_iso not in text:
        raise ValueError(f"{path}: expected datetime {old_iso!r} not found in content")
    text = text.replace(old_iso, new_iso)

    old_prefix = old_dt.strftime("%Y%m%d_%H%M%S")
    new_prefix = new_dt.strftime("%Y%m%d_%H%M%S")
    text = text.replace(old_prefix, new_prefix)

    out_path.write_text(text, encoding="utf-8")
    return out_path


def process_clock_shift(
    delta: timedelta,
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
) -> list[Path]:
    """Correct a flat camera clock drift/error affecting a batch of files.

    For each file in `in_dir` with a standardized name
    (``YYYYMMDD_HHMMSS...``), adds `delta` to the datetime encoded in the
    file name, then writes a copy to `out_dir` with:

    - the file name's ``YYYYMMDD_HHMMSS`` prefix replaced by the corrected
      datetime (see :func:`~mediautils.file_name.replace_datetime_prefix`);
    - the metadata timestamp set to the corrected datetime, dispatched by
      file extension like :func:`set_time` (EXIF ``datetime_original`` for
      images and RAW files, MP4/MOV ``creation_time`` for videos);
      ``.xmp`` sidecar files (e.g. Lightroom edits) are instead handled by
      shifting their embedded dates via text substitution, see
      :func:`_shift_xmp_datetimes`;
    - the copy's filesystem creation, modification and access times set to
      match (see :func:`~mediautils.image.set_file_creation_time`), since
      tools like FastStone Image Viewer's rename feature read those instead
      of the EXIF metadata, and to give every tool the best chance of
      sorting the files chronologically.

    Unlike :func:`process_timezone_shift`, this does not touch any EXIF
    timezone/offset tag: it corrects a clock drift (the camera's wall-clock
    reading was wrong), not a timezone error, so an existing offset tag is
    still correct and is left untouched.

    Parameters
    ----------
    delta : timedelta
        Amount to add to every timestamp. Use a negative value if the
        camera's clock was ahead of the real time.
    in_dir : str | os.PathLike
        Directory containing the files to correct, with standardized names.
    out_dir : str | os.PathLike
        Directory where corrected copies are saved.

    Returns
    -------
    list[Path]
        Paths to the output files.
    """
    in_dir = Path(in_dir)
    results = []
    for path in sorted(p for p in in_dir.iterdir() if p.is_file()):
        print(path.name)
        old_dt = file_name_to_datetime(path)
        new_dt = old_dt + delta
        out_name = replace_datetime_prefix(path, new_dt)
        if path.suffix.lower() in _XMP_EXTENSIONS:
            result = _shift_xmp_datetimes(
                path, old_dt, new_dt, out_dir=out_dir, out_name=out_name
            )
        else:
            result = set_time(path, new_dt, out_dir=out_dir, out_name=out_name)
        set_file_creation_time(result, new_dt)
        mod_time = new_dt.timestamp()
        os.utime(result, (mod_time, mod_time))
        results.append(result)
    print("Done!")
    return results
