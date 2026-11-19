"""Running exiftool safely with any file path."""

import os
import subprocess
import tempfile


def run_exiftool(args: list[str]) -> str:
    """Run exiftool with its arguments in a UTF-8 argument file; return its output.

    On Windows, exiftool misreads the paths given on the command line that hold
    characters outside the ANSI code page (e.g. « → »): every argument goes through
    an argument file instead (``-charset filename=utf8 -@ file``), one per line.

    Parameters
    ----------
    args : list[str]
        Arguments of exiftool (options, tag assignments, file paths), without
        ``exiftool`` itself. None may contain a line break.

    Returns
    -------
    str
        The standard output of exiftool, decoded as UTF-8.

    Raises
    ------
    subprocess.CalledProcessError
        If exiftool exits with a non-zero status.
    """
    if any("\n" in arg or "\r" in arg for arg in args):
        raise ValueError(f"exiftool argument with a line break: {args}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".args", delete=False
    ) as f:
        f.write("\n".join(args) + "\n")
    try:
        return subprocess.run(
            ["exiftool", "-charset", "filename=utf8", "-@", f.name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    finally:
        os.unlink(f.name)
