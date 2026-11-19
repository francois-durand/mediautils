import shutil
import subprocess
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from mediautils import (
    mirror_video,
    set_all_dates_video,
    set_time_raw,
    turn_video,
    write_precise_start,
)
from mediautils._exiftool import run_exiftool

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


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool is needed")
def test_turn_video(tmp_path):
    src = tmp_path / "lumix → tourner.mp4"
    # white on the left, black on the right; some sound; a date
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"color=white:s={W // 2}x{H}:d=0.4:r=25",
            "-f", "lavfi", "-i", f"color=black:s={W // 2}x{H}:d=0.4:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
            "-filter_complex", "[0][1]hstack[v]", "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-metadata", "creation_time=2026-06-12T18:43:08Z",
            str(src),
        ],
        check=True,
    )  # fmt: skip
    dst = tmp_path / "tournée → 90.mp4"
    with pytest.raises(ValueError):
        turn_video(src, dst, 45)
    assert turn_video(src, dst) == dst
    with pytest.raises(FileExistsError):
        turn_video(src, dst)

    def shown(path):  # first picture as a player shows it (rotation applied), W tall
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
        return np.frombuffer(out, np.uint8).reshape(W, H)

    a = shown(dst)  # turned clockwise: the white left half is now on top
    assert a[: W // 2 - 2].mean() > 220 and a[W // 2 + 2 :].mean() < 30
    # the turns add up: 90 + 180 = 270, white at the bottom
    dst2 = turn_video(dst, tmp_path / "270.mp4", 180)
    b = shown(dst2)
    assert b[: W // 2 - 2].mean() < 30 and b[W // 2 + 2 :].mean() > 220

    # the streams themselves are untouched (no re-encoding), the date kept
    def packets(path):
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0", "-c", "copy",
             "-f", "streamhash", "-hash", "md5", "-"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip

    assert packets(dst) == packets(src)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,width,height:format_tags=creation_time",
            "-of", "csv=p=0", str(dst),
        ],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert f"video,{W},{H}" in probe and "audio" in probe
    assert "2026-06-12T18:43:08" in probe


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool is needed")
def test_dates_paths_outside_ansi(tmp_path):
    """« → » in the folders (Windows: outside the ANSI code page) does not stop exiftool."""
    src_dir = tmp_path / "01_À renommer et-ou retagguer (→ 02)"
    out_dir = tmp_path / "02_Datées (→ 03)"
    src_dir.mkdir()
    src = src_dir / "VID → 20190924_162534.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.4",
            "-metadata", "creation_time=2001-01-01T00:00:00Z", str(src),
        ],
        check=True,
    )  # fmt: skip
    dt = datetime(2019, 9, 24, 16, 25, 34)

    out, unwritable = set_all_dates_video(src, dt, out_dir)
    assert out == out_dir / src.name and unwritable == {}
    stored = run_exiftool(
        ["-s3", "-api", "QuickTimeUTC=0", "-QuickTime:CreateDate", str(out)]
    )
    assert stored.strip() == "2019:09:24 14:25:34"  # 16:25:34 in Paris, summer time
    assert datetime.fromtimestamp(out.stat().st_mtime) == dt

    write_precise_start(out, dt.replace(microsecond=668000), frame_rate=25)
    shot = run_exiftool(["-s3", "-XMP-xmpDM:ShotDate", str(out)])
    assert shot.strip() == "2019:09:24 16:25:34.668+02:00"

    jpg = src_dir / "IMG → 1.jpg"
    Image.new("RGB", (8, 8)).save(jpg)
    raw = set_time_raw(jpg, dt, out_dir, offset="+02:00")
    original = run_exiftool(
        ["-s3", "-DateTimeOriginal", "-OffsetTimeOriginal", str(raw)]
    )
    assert original.split() == ["2019:09:24", "16:25:34", "+02:00"]
