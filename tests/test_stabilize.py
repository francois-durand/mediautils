import math
import shutil
import subprocess

import numpy as np
import pytest

from mediautils import camera_motion, stabilize_handheld, stabilized_sheet

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is needed"
)

W, H, N = 320, 180, 30


def _sim(dx, dy, deg):
    """Similarity: rotation of `deg` around the centre of the picture, then a shift."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = W / 2, H / 2
    return np.array(
        [
            [c, -s, cx - c * cx + s * cy + dx],
            [s, c, cy - s * cx - c * cy + dy],
            [0, 0, 1],
        ]
    )


def _moves():
    return [
        _sim(12 * math.sin(n / 5), 6 * math.sin(n / 7), 0.8 * math.sin(n / 6))
        for n in range(N)
    ]


@pytest.fixture(scope="module")
def handheld(tmp_path_factory):
    """A scene filmed by hand: picture n shows the scene through the motion n (+ some sound)."""
    path = tmp_path_factory.mktemp("stab") / "main levée.mp4"
    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.uniform(0, 255, (H + 120, W + 160)), (0, 0), 1.5)
    scene = cv2.normalize(scene, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    shift = np.array(
        [[1, 0, 80], [0, 1, 60], [0, 0, 1.0]]
    )  # picture 0 = centre of the scene
    frames = []
    for m in _moves():
        grey = cv2.warpAffine(
            scene, (shift @ m)[:2], (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
        )
        frames.append(
            cv2.cvtColor(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2YUV_I420)
        )
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{W}x{H}",
            "-r",
            "25",
            "-i",
            "-",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.2",
            "-shortest",
            "-c:v",
            "libx264",
            "-crf",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-metadata",
            "creation_time=2025-11-18T16:54:54Z",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    enc.stdin.write(b"".join(f.tobytes() for f in frames))
    enc.stdin.close()
    assert enc.wait() == 0
    return path


def _grey_frames(path):
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
    return np.frombuffer(out, np.uint8).reshape(-1, H, W).astype(np.float32)


def test_camera_motion(handheld):
    mats = camera_motion(handheld, smooth=0)
    assert mats.shape == (N, 3, 3)
    # picture n -> picture 0 = (move 0)^-1 . (move n): compare where points of the picture go
    pts = np.array([[40, 30], [280, 30], [160, 150]], float)
    m0 = np.linalg.inv(_moves()[0])
    for n, true in enumerate(_moves()):
        expected = (m0 @ true)[:2, :2] @ pts.T + (m0 @ true)[:2, 2:3]
        got = mats[n][:2, :2] @ pts.T + mats[n][:2, 2:3]
        assert np.abs(got - expected).max() < 0.5, n


def test_stabilize_handheld(handheld, tmp_path):
    with pytest.raises(ValueError, match="out of picture"):
        stabilize_handheld(handheld, tmp_path / "x.mp4", frame=(0, 0, W), smooth=0)
    dst = tmp_path / "stabilisée.mp4"
    res = stabilize_handheld(handheld, dst, smooth=0)
    assert res["frames"] == N and res["margin_px"][0] >= 2 and res["zoom"] > 1
    with pytest.raises(FileExistsError):
        stabilize_handheld(handheld, dst)
    # the scene does not move any more (it moved up to 12 px)
    f = _grey_frames(dst)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    moves = [cv2.phaseCorrelate(f[0], f[n], win)[0] for n in range(1, N)]
    assert np.abs(moves).max() < 0.6
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format_tags=creation_time",
            "-of",
            "csv=p=0",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "audio" in probe and "2025-11-18T16:54:54" in probe


def test_camera_motion_zone(handheld):
    # only the points of the left half of the first picture: the same motion
    mats = camera_motion(handheld, smooth=0, zone=(0, 0, W // 2, H))
    pts = np.array([[40, 30], [280, 30], [160, 150]], float)
    m0 = np.linalg.inv(_moves()[0])
    for n, true in enumerate(_moves()):
        expected = (m0 @ true)[:2, :2] @ pts.T + (m0 @ true)[:2, 2:3]
        got = mats[n][:2, :2] @ pts.T + mats[n][:2, 2:3]
        assert np.abs(got - expected).max() < 0.6, n
    with pytest.raises(ValueError, match="too few details"):
        camera_motion(handheld, zone=(0, 0, 4, 4))


def test_stabilize_handheld_mirror(handheld, tmp_path):
    motion = camera_motion(handheld, smooth=0)
    frame = (40, 30, 240)
    dst = tmp_path / "miroir.mp4"
    res = stabilize_handheld(handheld, dst, frame=frame, motion=motion, mirror=True)
    assert res["frames"] == N
    f = _grey_frames(dst)
    # first picture: the frame of the mirrored first picture of the source, enlarged
    src0 = _grey_frames(handheld)[0][:, ::-1]
    x0, y0, fw = frame
    fh = fw * H // W
    want = cv2.resize(
        np.ascontiguousarray(src0[y0 : y0 + fh, x0 : x0 + fw]),
        (W, H),
        interpolation=cv2.INTER_LANCZOS4,
    )
    assert np.abs(f[0] - want)[10:-10, 10:-10].mean() < 6
    assert np.abs(f[0] - want[:, ::-1])[10:-10, 10:-10].mean() > 20
    # and the scene does not move any more
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    moves = [cv2.phaseCorrelate(f[0], f[n], win)[0] for n in range(1, N)]
    assert np.abs(moves).max() < 0.8


def test_stabilize_handheld_follow(handheld, tmp_path):
    motion = camera_motion(handheld, smooth=0)
    fixed = stabilize_handheld(handheld, tmp_path / "fixe.mp4", motion=motion)
    # σ = 0.1 s = 2.5 pictures: the camera follows the moves of the video (periods of
    # 30-45 pictures), so it crops much less, and the picture still moves
    soft = stabilize_handheld(
        handheld, tmp_path / "souple.mp4", motion=motion, follow=0.1
    )
    assert soft["zoom"] < fixed["zoom"] - 0.1
    f = _grey_frames(tmp_path / "souple.mp4")
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    moves = [cv2.phaseCorrelate(f[0], f[n], win)[0] for n in range(1, N)]
    assert np.abs(moves).max() > 3


def test_stabilized_sheet(handheld, tmp_path):
    out = tmp_path / "planche → cadre.jpg"
    assert stabilized_sheet(handheld, out, frame=(40, 30, 240), step=0.4) == out
    sheet = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_COLOR)
    assert sheet.shape == (
        360,
        4 * 640,
        3,
    )  # 1.2 s every 0.4 s: 3 pictures + 1 empty, 640 wide
