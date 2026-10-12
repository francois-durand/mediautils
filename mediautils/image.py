import os
import shutil
import subprocess
import sys
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


def get_time_image(path: str | os.PathLike) -> datetime:
    """Read the ``datetime_original`` EXIF tag of an image.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the image.

    Returns
    -------
    datetime
        The datetime stored in the ``datetime_original`` EXIF tag, naive
        (no timezone attached, regardless of whether an ``offset_time_original``
        tag is present).
    """
    image = ExifImage(path)
    return datetime.strptime(image.datetime_original, DATETIME_STR_FORMAT)


def get_offset_image(path: str | os.PathLike) -> str | None:
    """Read the ``offset_time_original`` EXIF tag of an image.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the image.

    Returns
    -------
    str | None
        The UTC offset stored in the ``offset_time_original`` EXIF tag
        (e.g. ``"+09:00"``), or ``None`` if the tag is absent.
    """
    image = ExifImage(path)
    return getattr(image, "offset_time_original", None)


def set_file_creation_time(path: str | os.PathLike, dt: datetime) -> None:
    """Set a file's Windows filesystem creation-time metadata.

    Unlike EXIF timestamps, Windows filesystem timestamps carry no timezone:
    they are stored as an absolute instant and redisplayed converted to
    whatever timezone the machine currently uses. `dt` is interpreted as
    local wall-clock time on the current machine (exactly like
    :meth:`datetime.timestamp`), so that a tool reading the creation time
    back on this same machine (e.g. Windows Explorer or FastStone Image
    Viewer) redisplays the same digits.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the file.
    dt : datetime
        The datetime to set as creation time, interpreted in the machine's
        current local timezone.

    Raises
    ------
    NotImplementedError
        If not running on Windows.
    """
    if sys.platform != "win32":
        raise NotImplementedError(
            "Setting file creation time is only supported on Windows."
        )

    import pywintypes
    import win32con
    import win32file

    handle = win32file.CreateFile(
        str(path),
        win32con.GENERIC_WRITE,
        win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None,
    )
    try:
        win32file.SetFileTime(handle, pywintypes.Time(dt), None, None)
    finally:
        handle.Close()


def set_time_image(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
    offset: str | None = None,
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
    offset : str | None
        UTC offset to write to the ``offset_time_original`` EXIF tag, in
        ``"+HH:MM"``/``"-HH:MM"`` format (e.g. ``"+09:00"``). If ``None``,
        the tag is left untouched.

    Returns
    -------
    Path
        Path to the output file.

    Notes
    -----
    If the ``exif`` package cannot parse the file, falls back on
    :func:`set_time_raw` (requires ``exiftool`` on `PATH`).
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        image = ExifImage(path)
    except Exception:
        # The ``exif`` package chokes on some JPEGs (e.g. a non-EXIF APP1
        # segment placed first); exiftool copes with them.
        return set_time_raw(path, dt, out_dir=out_dir, out_name=out_name, offset=offset)
    image.datetime_original = dt.strftime(DATETIME_STR_FORMAT)
    if offset is not None:
        image.offset_time_original = offset
    out_path = out_dir / (out_name or path.name)
    with open(out_path, "wb") as f:
        f.write(image.get_file())
    return out_path


def set_time_raw(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
    offset: str | None = None,
) -> Path:
    """Write a copy of a RAW image with ``DateTimeOriginal`` set to `dt`.

    Camera RAW formats (e.g. Panasonic ``.rw2``) are TIFF-based, not
    JPEG-APP1-based, so the :class:`exif.Image` interface used by
    :func:`set_time_image` cannot parse them. This uses the external
    `exiftool <https://exiftool.org/>`_ command-line tool instead, which
    must be on `PATH`. Any other metadata tag is left untouched.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source RAW image.
    dt : datetime
        The datetime to write into the ``DateTimeOriginal`` metadata tag.
    out_dir : str | os.PathLike
        Directory where the modified copy is saved. Created if it does not exist.
    out_name : str | None
        Output file name. If ``None``, the original file name is kept.
    offset : str | None
        UTC offset to write to the ``OffsetTimeOriginal`` metadata tag, in
        ``"+HH:MM"``/``"-HH:MM"`` format (e.g. ``"+09:00"``). If ``None``,
        the tag is left untouched.

    Returns
    -------
    Path
        Path to the output file.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)
    shutil.copy2(path, out_path)
    args = [
        "exiftool",
        "-overwrite_original",
        f"-DateTimeOriginal={dt.strftime('%Y:%m:%d %H:%M:%S')}",
    ]
    if offset is not None:
        args.append(f"-OffsetTimeOriginal={offset}")
    args.append(str(out_path))
    subprocess.run(args, check=True, capture_output=True)
    return out_path


def get_all_image_files(
    root_dir: str | os.PathLike,
    exclude: list[str] | None = None,
) -> list[Path]:
    """List all JPEG and PNG files under `root_dir` recursively.

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
    image_files: list[Path] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(pattern in dirpath for pattern in exclude):
            continue
        for name in filenames:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(Path(dirpath) / name)
    return image_files


def get_orientation(path: str | os.PathLike) -> str:
    """Detect whether an image is portrait or landscape.

    Takes EXIF orientation into account (rotation 90° or 270°).
    Square images are reported as ``"landscape"``.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a JPEG or PNG image.

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


def dispatch_images_by_orientation(
    src_dir: str | os.PathLike = "in",
    dest_portrait: str | os.PathLike = "out/v",
    dest_landscape: str | os.PathLike = "out/h",
    max_dim: int = 1024,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Dispatch JPEG and PNG images by orientation and resize them.

    Collects all JPEG and PNG files under `src_dir` (recursively), detects
    each image's orientation, copies it to `dest_portrait` or
    `dest_landscape`, and resizes it if its largest dimension exceeds
    `max_dim`.

    Parameters
    ----------
    src_dir : str | os.PathLike
        Source directory to scan for image files.
    dest_portrait : str | os.PathLike
        Output directory for portrait images.
    dest_landscape : str | os.PathLike
        Output directory for landscape images.
    max_dim : int
        Maximum size in pixels for the largest dimension.
    exclude : list[str] | None
        Exclusion patterns passed to :func:`get_all_image_files`.

    Returns
    -------
    list[Path]
        Paths to the output files.
    """
    image_files = get_all_image_files(src_dir, exclude=exclude)
    results: list[Path] = []

    Path(dest_portrait).mkdir(parents=True, exist_ok=True)
    Path(dest_landscape).mkdir(parents=True, exist_ok=True)

    for file_path in image_files:
        orientation = get_orientation(file_path)
        dest_dir = dest_portrait if orientation == "portrait" else dest_landscape
        clean_name = Path(file_path).name.replace(":", "_")
        try:
            result = resize_image(
                file_path, max_dim=max_dim, out_dir=dest_dir, out_name=clean_name
            )
        except OSError:
            dest_dir_path = Path(dest_dir)
            dest_dir_path.mkdir(parents=True, exist_ok=True)
            result = dest_dir_path / clean_name
            shutil.copy2(file_path, result)
        results.append(result)

    return results
