import os
from datetime import datetime, time
from pathlib import Path
from shutil import copy2

from mediautils.file_name import (
    _date_prefix_to_date,
    file_name_to_datetime,
    wa_file_name_to_date,
)
from mediautils.image import set_time_image
from mediautils.video import set_time_video

_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
_COPY_ONLY_EXTENSIONS = {".png"}
_VIDEO_EXTENSIONS = {".mp4", ".mov"}


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


def process_standard_files(
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


def process_wa_files(
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


def process_directory_files(
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
