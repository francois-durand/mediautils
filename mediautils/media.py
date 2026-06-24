import os
from datetime import datetime, time
from pathlib import Path

from mediautils.file_name import wa_file_name_to_date
from mediautils.image import set_time_image
from mediautils.video import set_time_video

_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
_VIDEO_EXTENSIONS = {".mp4"}


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
    raise ValueError(f"Unsupported file extension: {ext}")


def process_wa_files(
    in_dir: str | os.PathLike = "in",
    out_dir: str | os.PathLike = "out",
) -> list[Path]:
    """Process WhatsApp media files: set timestamps and rename.

    For each file in `in_dir`, extracts the date from the WhatsApp file name,
    assigns a time starting at 23:50:01 (incrementing by one second per file
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

    previous_date = None
    counter = 0
    results = []

    for path, d in entries:
        if d != previous_date:
            previous_date = d
            counter = 1
        else:
            counter += 1

        second = counter % 60
        minute = 50 + (counter - second) // 60
        dt = datetime.combine(d, time(hour=23, minute=minute, second=second))
        out_name = f"{dt.strftime('%Y%m%d_%H%M%S')}_{path.name}"

        result = set_time(path, dt, out_dir=out_dir, out_name=out_name)
        results.append(result)

    return results
