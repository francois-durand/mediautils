import os
import re
from datetime import datetime
from pathlib import Path

_PATTERN = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


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
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


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
