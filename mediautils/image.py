import os
import shutil
from datetime import datetime
from pathlib import Path

from exif import DATETIME_STR_FORMAT, Image as ExifImage
from PIL import Image as PILImage

_EXIF_ORIENTATION_TAG = 0x0112

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "-mémoire",
    "très longue",
    "Allégeance",
    "Visites d'appart",
    "2017 05 03 Visite",
    "2017 05 14 Pré-état des lieux",
    "2011 05 30 Ma chérie",
    "Etat des lieux",
    "Paris Picpus, dégât des eaux",
]


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
    image = ExifImage(path)
    image.datetime_original = dt.strftime(DATETIME_STR_FORMAT)
    out_path = out_dir / (out_name or path.name)
    with open(out_path, "wb") as f:
        f.write(image.get_file())
    return out_path


def get_all_jpeg_files(
    root_dir: str | os.PathLike,
    exclude: list[str] | None = None,
) -> list[Path]:
    """List all JPEG files under `root_dir` recursively.

    Parameters
    ----------
    root_dir : str | os.PathLike
        Root directory to search.
    exclude : list[str] | None
        Substrings to exclude: any directory whose path contains one of these
        strings is skipped. If ``None``, :data:`DEFAULT_EXCLUDE_PATTERNS` is used.
        Pass an empty list to disable exclusions.

    Returns
    -------
    list[Path]
    """
    if exclude is None:
        exclude = DEFAULT_EXCLUDE_PATTERNS
    root_dir = Path(root_dir)
    jpeg_files: list[Path] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(pattern in dirpath for pattern in exclude):
            continue
        for name in filenames:
            if name.lower().endswith((".jpg", ".jpeg")):
                jpeg_files.append(Path(dirpath) / name)
    return jpeg_files


def get_orientation(path: str | os.PathLike) -> str:
    """Detect whether an image is portrait or landscape.

    Takes EXIF orientation into account (rotation 90° or 270°).
    Square images are reported as ``"landscape"``.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a JPEG image.

    Returns
    -------
    str
        ``"portrait"`` or ``"landscape"``.
    """
    with PILImage.open(path) as img:
        width, height = img.size
        exif = img.getexif()
        if exif.get(_EXIF_ORIENTATION_TAG) in (6, 8):
            width, height = height, width
    return "portrait" if height > width else "landscape"


def resize_image(
    path: str | os.PathLike,
    max_dim: int = 1024,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a copy of an image, resized if its largest dimension exceeds `max_dim`.

    Images with EXIF orientation 6 or 8 (rotated 90°/270°) are copied without
    resizing to avoid corrupting the orientation metadata.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source image.
    max_dim : int
        Maximum size in pixels for the largest dimension.
    out_dir : str | os.PathLike
        Directory where the output file is saved. Created if it does not exist.
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

    with PILImage.open(path) as img:
        exif = img.getexif()
        orientation = exif.get(_EXIF_ORIENTATION_TAG)
        width, height = img.size

        if orientation not in (6, 8) and max(width, height) > max_dim:
            if width >= height:
                new_width = max_dim
                new_height = round(height * max_dim / width)
            else:
                new_height = max_dim
                new_width = round(width * max_dim / height)
            img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            img.save(out_path, quality=95)
            return out_path

    shutil.copy2(path, out_path)
    return out_path


def process_images(
    src_dir: str | os.PathLike,
    dest_portrait: str | os.PathLike,
    dest_landscape: str | os.PathLike,
    max_dim: int = 1024,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Dispatch JPEG images by orientation and resize them.

    Collects all JPEG files under `src_dir` (recursively), detects each
    image's orientation, copies it to `dest_portrait` or `dest_landscape`,
    and resizes it if its largest dimension exceeds `max_dim`.

    Parameters
    ----------
    src_dir : str | os.PathLike
        Source directory to scan for JPEG files.
    dest_portrait : str | os.PathLike
        Output directory for portrait images.
    dest_landscape : str | os.PathLike
        Output directory for landscape images.
    max_dim : int
        Maximum size in pixels for the largest dimension.
    exclude : list[str] | None
        Exclusion patterns passed to :func:`get_all_jpeg_files`.

    Returns
    -------
    list[Path]
        Paths to the output files.
    """
    jpeg_files = get_all_jpeg_files(src_dir, exclude=exclude)
    results: list[Path] = []

    for file_path in jpeg_files:
        orientation = get_orientation(file_path)
        dest_dir = dest_portrait if orientation == "portrait" else dest_landscape
        clean_name = Path(file_path).name.replace(":", "_")
        try:
            result = resize_image(file_path, max_dim=max_dim, out_dir=dest_dir, out_name=clean_name)
        except OSError:
            dest_dir_path = Path(dest_dir)
            dest_dir_path.mkdir(parents=True, exist_ok=True)
            result = dest_dir_path / clean_name
            shutil.copy2(file_path, result)
        results.append(result)

    return results
