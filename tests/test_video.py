import shutil
import subprocess

import numpy as np
import pytest

from mediautils import mirror_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is needed"
)

W, H = 64, 36


def _grey(path):
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W)


def test_mirror_video(tmp_path):
    src = tmp_path / "selfie → miroir.mp4"
    # white on the left, black on the right; some sound; a date
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"color=white:s={W // 2}x{H}:d=0.4:r=25",
            "-f", "lavfi", "-i", f"color=black:s={W // 2}x{H}:d=0.4:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
            "-filter_complex", "[0][1]hstack[v]", "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-metadata", "creation_time=2026-02-11T18:10:26Z",
            str(src),
        ],
        check=True,
    )  # fmt: skip
    dst = tmp_path / "miroir.mp4"
    assert mirror_video(src, dst) == dst
    with pytest.raises(FileExistsError):
        mirror_video(src, dst)
    a, b = _grey(src), _grey(dst)
    assert a.shape == b.shape
    assert b[:, :, : W // 2 - 2].mean() < 30 and b[:, :, W // 2 + 2 :].mean() > 220
    assert np.abs(b.astype(int) - a[:, :, ::-1]).mean() < 3
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type:format_tags=creation_time",
            "-of", "csv=p=0", str(dst),
        ],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert "audio" in probe and "2026-02-11T18:10:26" in probe
