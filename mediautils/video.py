import os
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import copy2


def set_time_video(
    path: str | os.PathLike,
    dt: datetime,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a copy of a video with its timestamps set to `dt`.

    Sets the filesystem modification/access times and the MP4 container
    ``creation_time`` metadata (via ffmpeg).

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source video.
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
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)

    copy2(path, out_path)

    mod_time = dt.timestamp()
    os.utime(out_path, (mod_time, mod_time))

    metadata_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    tmp_path = out_path.with_stem(out_path.stem + "_tmp")
    subprocess.run(
        [
            "ffmpeg", "-i", str(out_path),
            "-metadata", f"creation_time={metadata_date}",
            "-codec", "copy", "-y", str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    tmp_path.replace(out_path)

    return out_path


def rotate_video(
    path: str | os.PathLike,
    angle: int = 90,
    out_dir: str | os.PathLike = "out",
    out_name: str | None = None,
) -> Path:
    """Write a rotated copy of a video.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the source video.
    angle : int
        Rotation angle in degrees (90, 180, or 270).
    out_dir : str | os.PathLike
        Directory where the rotated video is saved. Created if it does not exist.
    out_name : str | None
        Output file name. If ``None``, the original file name is kept.

    Returns
    -------
    Path
        Path to the output file.
    """
    from moviepy import VideoFileClip

    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or path.name)

    clip = VideoFileClip(str(path))
    rotated_clip = clip.rotated(angle)
    rotated_clip.write_videofile(str(out_path), codec="libx264")
    rotated_clip.close()
    clip.close()

    return out_path
