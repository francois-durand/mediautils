import os
import re
from datetime import date, datetime
from pathlib import Path

_PATTERN = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
_DATE_PREFIX_PATTERN = re.compile(r"^(\d{4})(\d{2})(\d{2})")
_WA_PATTERN = re.compile(r"^(?:IMG|VID)-(\d{4})(\d{2})(\d{2})-")

# Naming conventions of the sources met so far, other than the standard
# ``YYYYMMDD_HHMMSS...``. Each regex is matched against the file stem and must
# capture year, month, day, hour, minute, second (in that order). To support a
# new source, add one line here (and a doctest to :func:`standardize_name`).
# Standardized names whose prefix must be recomputed: an earlier renaming took
# the end of the recording (from the metadata) while the phone's own name,
# kept after the prefix, gives its start in the phone's local time.
_RESTAMP_PATTERN = re.compile(
    r"^\d{8}_\d{6}_(VID(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:_.*)?)$"
)

_KNOWN_FORMATS: list[tuple[str, re.Pattern]] = [
    (
        "Android phone video (VID_YYYYMMDD_HHMMSS[mmm])",
        re.compile(
            r"^VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:\d{3})?(?:_.*)?$"
        ),
    ),
]


def _file_name_to_datetime(name: str) -> datetime:
    """Extract a datetime from a file name string.

    Parameters
    ----------
    name : str
        A file name starting with ``YYYYMMDD_HHMMSS``.

    Returns
    -------
    datetime

    Raises
    ------
    ValueError
        If `name` does not match the expected format.

    Examples
    --------
        >>> _file_name_to_datetime("20240131_181253_blabla")
        datetime.datetime(2024, 1, 31, 18, 12, 53)

        >>> _file_name_to_datetime("20240131_181253")
        datetime.datetime(2024, 1, 31, 18, 12, 53)
    """
    match = _PATTERN.match(name)
    if not match:
        raise ValueError(
            f"File name does not match expected format 'YYYYMMDD_HHMMSS...': {name}"
        )
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    return datetime(
        year=year, month=month, day=day, hour=hour, minute=minute, second=second
    )


def file_name_to_datetime(path: str | os.PathLike) -> datetime:
    """Extract a datetime from a file path based on its name.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a file whose stem starts with ``YYYYMMDD_HHMMSS``.

    Returns
    -------
    datetime

    Raises
    ------
    ValueError
        If the file name does not match the expected format.

    Examples
    --------
        >>> file_name_to_datetime("20240131_181253_blabla.jpg")
        datetime.datetime(2024, 1, 31, 18, 12, 53)

        >>> file_name_to_datetime("some/dir/20250101_120000.png")
        datetime.datetime(2025, 1, 1, 12, 0)
    """
    return _file_name_to_datetime(Path(path).stem)


def _replace_datetime_prefix(name: str, dt: datetime) -> str:
    """Replace the ``YYYYMMDD_HHMMSS`` prefix of a file name string with `dt`.

    Parameters
    ----------
    name : str
        A file name starting with ``YYYYMMDD_HHMMSS``.
    dt : datetime
        The datetime to encode as the new prefix.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If `name` does not match the expected format.

    Examples
    --------
        >>> _replace_datetime_prefix(
        ...     "20240131_181253_blabla", datetime(2024, 1, 31, 18, 27, 53)
        ... )
        '20240131_182753_blabla'

        >>> _replace_datetime_prefix("20240131_181253.jpg", datetime(2024, 1, 31, 18, 12, 38))
        '20240131_181238.jpg'
    """
    match = _PATTERN.match(name)
    if not match:
        raise ValueError(
            f"File name does not match expected format 'YYYYMMDD_HHMMSS...': {name}"
        )
    return dt.strftime("%Y%m%d_%H%M%S") + name[match.end() :]


def replace_datetime_prefix(path: str | os.PathLike, dt: datetime) -> str:
    """Replace the ``YYYYMMDD_HHMMSS`` prefix of a file's name with `dt`.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a file whose name starts with ``YYYYMMDD_HHMMSS``.
    dt : datetime
        The datetime to encode as the new prefix.

    Returns
    -------
    str
        The corrected file name (not a path).

    Raises
    ------
    ValueError
        If the file name does not match the expected format.

    Examples
    --------
        >>> replace_datetime_prefix(
        ...     "20240131_181253_blabla.jpg", datetime(2024, 1, 31, 18, 27, 53)
        ... )
        '20240131_182753_blabla.jpg'

        >>> replace_datetime_prefix(
        ...     "some/dir/20250101_120000.png", datetime(2025, 1, 1, 11, 45, 0)
        ... )
        '20250101_114500.png'
    """
    return _replace_datetime_prefix(Path(path).name, dt)


def standardize_name(path: str | os.PathLike) -> tuple[str, datetime]:
    """Compute the standardized name of a file and the datetime it encodes.

    A name that is already standardized (``YYYYMMDD_HHMMSS...``) is kept as
    is, with one exception: ``YYYYMMDD_HHMMSS_VIDyyyymmddhhmmss...``, whose
    prefix is replaced by the datetime of the phone's own name (start of the
    recording, local time where it was filmed). A name following one of the
    known source conventions is prefixed with ``YYYYMMDD_HHMMSS_``, keeping the
    original name in full so that it is never lost (even if the date and time
    then appear twice).

    Parameters
    ----------
    path : str | os.PathLike
        Path to the file.

    Returns
    -------
    tuple[str, datetime]
        The standardized file name (not a path) and the datetime it encodes
        (naive, local wall-clock time).

    Raises
    ------
    ValueError
        If the file name follows no known convention.

    Examples
    --------
    Already standardized: unchanged.

        >>> standardize_name("20240131_181253_blabla.mp4")
        ('20240131_181253_blabla.mp4', datetime.datetime(2024, 1, 31, 18, 12, 53))

    Prefix recomputed from the phone's own name (end of recording -> start;
    MGifier suffixes kept):

        >>> standardize_name("20251126_185311_VID20251126185303.mp4")
        ('20251126_185303_VID20251126185303.mp4', datetime.datetime(2025, 11, 26, 18, 53, 3))

        >>> standardize_name("20260611_125751_VID20260611195724_MG.mp4")
        ('20260611_195724_VID20260611195724_MG.mp4', datetime.datetime(2026, 6, 11, 19, 57, 24))

    Android phone video (milliseconds dropped from the datetime):

        >>> standardize_name("some/dir/VID_20190924_162534168.mp4")
        ('20190924_162534_VID_20190924_162534168.mp4', datetime.datetime(2019, 9, 24, 16, 25, 34))

        >>> standardize_name("VID_20190924_162534.mp4")
        ('20190924_162534_VID_20190924_162534.mp4', datetime.datetime(2019, 9, 24, 16, 25, 34))

    Unknown convention:

        >>> standardize_name("MVI_1234.MOV")
        Traceback (most recent call last):
        ...
        ValueError: Unknown file name convention: MVI_1234.MOV
    """
    path = Path(path)
    match = _RESTAMP_PATTERN.match(path.stem)
    if match:
        dt = datetime(*(int(g) for g in match.groups()[1:]))
        return f"{dt:%Y%m%d_%H%M%S}_{match.group(1)}{path.suffix}", dt
    if _PATTERN.match(path.stem):
        return path.name, _file_name_to_datetime(path.stem)
    for _, pattern in _KNOWN_FORMATS:
        match = pattern.match(path.stem)
        if match:
            year, month, day, hour, minute, second = (int(g) for g in match.groups())
            dt = datetime(year, month, day, hour, minute, second)
            return f"{dt:%Y%m%d_%H%M%S}_{path.name}", dt
    raise ValueError(f"Unknown file name convention: {path.name}")


def _date_prefix_to_date(name: str) -> date:
    """Extract a date from a string starting with ``YYYYMMDD``.

    Parameters
    ----------
    name : str
        A string starting with ``YYYYMMDD``.

    Returns
    -------
    date

    Raises
    ------
    ValueError
        If `name` does not start with 8 digits forming a valid date.

    Examples
    --------
        >>> _date_prefix_to_date("20240229blabla")
        datetime.date(2024, 2, 29)

        >>> _date_prefix_to_date("20240703")
        datetime.date(2024, 7, 3)
    """
    match = _DATE_PREFIX_PATTERN.match(name)
    if not match:
        raise ValueError(f"Name does not match expected format 'YYYYMMDD...': {name}")
    year, month, day = (int(g) for g in match.groups())
    return date(year=year, month=month, day=day)


def _wa_file_name_to_date(name: str) -> date:
    """Extract a date from a WhatsApp file name string.

    Parameters
    ----------
    name : str
        A file name matching ``IMG-YYYYMMDD-...`` or ``VID-YYYYMMDD-...``.

    Returns
    -------
    date

    Raises
    ------
    ValueError
        If `name` does not match the expected WhatsApp format.

    Examples
    --------
        >>> _wa_file_name_to_date("IMG-20251014-WA0010.jpg")
        datetime.date(2025, 10, 14)

        >>> _wa_file_name_to_date("VID-20251027-WA0001.mp4")
        datetime.date(2025, 10, 27)
    """
    match = _WA_PATTERN.match(name)
    if not match:
        raise ValueError(
            f"File name does not match expected WhatsApp format 'IMG-YYYYMMDD-...' "
            f"or 'VID-YYYYMMDD-...': {name}"
        )
    year, month, day = (int(g) for g in match.groups())
    return date(year=year, month=month, day=day)


def wa_file_name_to_date(path: str | os.PathLike) -> date:
    """Extract a date from a WhatsApp file path based on its name.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a file matching ``IMG-YYYYMMDD-...`` or ``VID-YYYYMMDD-...``.

    Returns
    -------
    date

    Raises
    ------
    ValueError
        If the file name does not match the expected WhatsApp format.

    Examples
    --------
        >>> wa_file_name_to_date("IMG-20251014-WA0010.jpg")
        datetime.date(2025, 10, 14)

        >>> wa_file_name_to_date("some/dir/VID-20251027-WA0001.mp4")
        datetime.date(2025, 10, 27)
    """
    return _wa_file_name_to_date(Path(path).name)
