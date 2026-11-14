import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from mediautils import (
    SyncProject,
    add_logo,
    add_markers,
    build_montage,
    compare_exports,
    edit_frames,
    logo_overlay,
    marker_time,
    montage_shots,
    montage_views,
    move_cut,
    read_markers,
    ruler_image,
    run_export_queue,
    rush_sheets,
)
from tests.test_montage import FPS, make_sync_project

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is needed"
)


@pytest.fixture
def montage(tmp_path):
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    shots = [(0, "A"), (10, "zL"), (20, "L"), (30, "A")]
    build_montage(
        P, shots, {"zL": ("L", 960, 540, 2.0)}, [], tmp_path / "m.mlt", cut_end=40.0
    )
    return tmp_path / "m.mlt"


def close(a, b):
    return abs(a - b) < 1.5 / FPS


def test_markers(tmp_path):
    p = make_sync_project(tmp_path / "p.mlt")
    add_markers(p, [(10.0, "5.1 Couplet"), (2.0, "1.1"), (12.0, "6")])
    assert [m[0] for m in read_markers(p)] == ["1.1", "5.1 Couplet", "6"]
    assert close(marker_time(p, "5.1"), 10.0)
    assert close(marker_time(p, "5"), 10.0)
    assert close(marker_time(p, "6.1"), 12.0)  # the 1 of bar 6, marked « 6 »
    assert close(marker_time(p, "00:00:01:30"), 1.5)  # timecode, frames at ~60 fps
    with pytest.raises(KeyError):
        marker_time(p, "7.1")


def test_edit_frames(montage, tmp_path):
    zoom = next(
        c.get("id")
        for c in ET.parse(montage).getroot()
        if c.find("property[@name='montage:shot']") is not None
        and c.find("property[@name='montage:shot']").text == "zL"
    )
    dst = edit_frames(
        montage,
        tmp_path / "e.mlt",
        [
            # over the cut zL | L of track V1: one shot (same source, continuous)
            (12.0, 25.0, (480, 270, 1440, 810), "centre"),
            (25.0, 28.0, zoom, None),  # the framing of another shot, its name kept
        ],
    )
    shots = montage_shots(dst)
    assert [s[2] for s in shots] == ["A", "zL", "centre", "zL", "L", "A"]
    assert close(shots[2][0], 12.0) and close(shots[2][1], 25.0)
    assert close(shots[3][0], 25.0) and close(shots[3][1], 28.0)
    assert close(shots[-1][1], 40.0)  # nothing else moved
    # the rectangle becomes a zoom x2 on the middle of the picture
    root = ET.parse(dst).getroot()
    rects = [
        f.find("property[@name='transition.rect']").text
        for f in root.iter("filter")
        if f.get("id", "").startswith("revise_affine")
    ]
    assert "-960.0 -540.0 3840.0 2160.0 1" in rects
    with pytest.raises(FileExistsError):
        edit_frames(montage, dst, [])
    with pytest.raises(ValueError):
        edit_frames(montage, tmp_path / "x.mlt", [(1, 2, (0, 0, 100, 100), None)])


def test_move_cut(montage, tmp_path):
    # the cut zL | L of track V1 (camera A is on V2)
    dst = move_cut(montage, tmp_path / "c.mlt", 20.0, 21.0)
    shots = montage_shots(dst)
    assert [s[2] for s in shots] == ["A", "zL", "L", "A"]
    assert close(shots[1][1], 21.0) and close(shots[2][0], 21.0)
    assert close(shots[2][1], 30.0) and close(shots[3][1], 40.0)
    with pytest.raises(ValueError):
        move_cut(montage, tmp_path / "d.mlt", 12.0, 13.0)  # no cut there


def test_logo_hides_nothing(montage, tmp_path):
    # a still picture on a top track (the logo) is not taken for a shot
    from PIL import Image

    Image.new("RGBA", (100, 50), (200, 20, 20, 255)).save(tmp_path / "logo.png")
    logo_overlay(tmp_path / "logo.png", tmp_path / "o.png", 1920, 1080, 0.1, 0.05, 0.02)
    add_logo(montage, tmp_path / "l.mlt", "o.png")
    assert montage_shots(tmp_path / "l.mlt") == montage_shots(montage)


def _video(path, seconds, colour):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color={colour}:s=320x180:r=25:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@NEEDS_FFMPEG
def test_pictures(tmp_path):
    P = SyncProject(make_sync_project(tmp_path / "p.mlt", ref_len=12.0, other_len=11.0))
    _video(tmp_path / "x_P1000001.MP4", 12, "red")
    _video(tmp_path / "y_MVI_0001.MOV", 11, "blue")
    build_montage(
        P,
        [(0, "A"), (5, "zL")],
        {"zL": ("L", 960, 540, 2.0)},
        [],
        tmp_path / "m.mlt",
        cut_end=10.0,
    )
    from PIL import Image

    v = montage_views(
        tmp_path / "m.mlt", [1.0, 7.0], tmp_path / "v.jpg", cols=2, width=320
    )
    im = Image.open(v).convert("RGB")
    assert im.size == (640, 180)
    assert im.getpixel((160, 120))[2] > 150  # blue camera
    assert im.getpixel((480, 120))[0] > 150  # red camera
    sheets = rush_sheets(
        tmp_path / "m.mlt", tmp_path / "s", step=2.0, cols=2, rows=2, width=160
    )
    assert len(sheets) == 2 and all(s.exists() for s in sheets)
    r = ruler_image(
        tmp_path / "m.mlt", 7.0, tmp_path / "r.jpg", [(480, 270, 1440, 810)], width=960
    )
    assert Image.open(r).size == (960, 540)


@NEEDS_FFMPEG
def test_compare_exports(tmp_path):
    _video(tmp_path / "a.mp4", 4, "gray")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(tmp_path / "a.mp4"),
            "-vf",
            "drawbox=x=iw-40:y=ih-30:w=30:h=20:color=white:t=fill",
            "-c:a",
            "copy",
            str(tmp_path / "b.mp4"),
        ],
        check=True,
    )
    c = compare_exports(tmp_path / "b.mp4", tmp_path / "a.mp4", sound=1.0)
    assert abs(c["duration_new"] - c["duration_old"]) < 0.1
    assert abs(c["level_new"] - c["level_old"]) < 0.5
    assert c["diff_outside"] < 3 < 8 < c["diff_corner"]


def test_run_export_queue(tmp_path):
    (tmp_path / "job.py").write_text(
        "import sys, pathlib\npathlib.Path(sys.argv[1]).write_text('ok')\n"
        "sys.exit(int(sys.argv[2]))\n",
        encoding="utf-8",
    )
    q = tmp_path / "queue.txt"
    q.write_text(
        "# a comment\njob.py a.txt 0\njob.py b.txt 1\nFIN\njob.py c.txt 0\n",
        encoding="utf-8",
    )
    run_export_queue(q, poll=0.01)
    assert q.read_text(encoding="utf-8").splitlines() == [
        "# a comment",
        "fait job.py a.txt 0",
        "ÉCHEC job.py b.txt 1",
        "FIN",
        "job.py c.txt 0",
    ]
    assert (tmp_path / "a.txt").exists() and not (tmp_path / "c.txt").exists()
    assert (tmp_path / "export_job_a.txt_0.log").exists()
