"""Maintenance of Shotcut projects (``.mlt`` files)."""

import os
import re
from pathlib import Path
from shutil import copy2
from xml.sax.saxutils import escape, unescape

# Properties naming a source file: its path, and the caption shown by Shotcut.
_FILE_PROPERTY = re.compile(
    r'(<property name="(?:resource|shotcut:caption)">)([^<]*)(</property>)'
)


def relink_project(
    path: str | os.PathLike,
    renames: dict[str, str] | None = None,
    backup: bool = True,
) -> list[tuple[str, str]]:
    """Make the links of a Shotcut project point to files next to it.

    Each source file referenced by the project (absolute or relative path) is
    replaced by its bare file name, i.e. a path relative to the project, if a
    file of that name (after applying `renames`) exists in the folder of the
    project. Other references (e.g. a music file elsewhere) are left as they
    are. Only these references change: in and out points, positions on the
    timeline and everything else are kept byte for byte. The file is modified
    in place.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the ``.mlt`` project.
    renames : dict[str, str] | None
        Old file name to new file name, for source files that were renamed.
    backup : bool
        If ``True``, the original project is first copied to ``<name>.mlt.bak``
        (unless such a backup already exists).

    Returns
    -------
    list[tuple[str, str]]
        The changed references, as (old, new).
    """
    path = Path(path)
    renames = renames or {}
    text = path.read_bytes().decode("utf-8")
    changes = []

    def relink(match: re.Match) -> str:
        value = unescape(match.group(2))
        leaf = re.split(r"[\\/]", value)[-1]
        new = renames.get(leaf, leaf)
        if not new or not (path.parent / new).is_file() or new == value:
            return match.group(0)
        changes.append((value, new))
        return match.group(1) + escape(new) + match.group(3)

    new_text = _FILE_PROPERTY.sub(relink, text)
    if changes:
        bak = path.with_name(path.name + ".bak")
        if backup and not bak.exists():
            copy2(path, bak)
        path.write_bytes(new_text.encode("utf-8"))
    return changes


def project_files(path: str | os.PathLike) -> list[str]:
    """List the source files referenced by a Shotcut project (``resource``).

    Returns
    -------
    list[str]
        The references, as written in the project (``"0"`` and similar
        generator values included).
    """
    text = Path(path).read_bytes().decode("utf-8")
    return [
        unescape(m.group(2))
        for m in _FILE_PROPERTY.finditer(text)
        if m.group(1).endswith('"resource">')
    ]
