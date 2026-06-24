import os
from datetime import datetime
from pathlib import Path

from exif import DATETIME_STR_FORMAT, Image


def set_time_image(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a copy of an image with ``datetime_original`` set to `dt`.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source image.
    dt : datetime
        The datetime to write into the EXIF metadata.
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
    image = Image(path)
    image.datetime_original = dt.strftime(DATETIME_STR_FORMAT)
    out_path = out_dir / (out_name or path.name)
    with open(out_path, "wb") as f:
        f.write(image.get_file())
    return out_path
