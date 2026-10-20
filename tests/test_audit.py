import numpy as np
import pytest

from mediautils import (
    SyncProject,
    audit_cuts,
    build_montage,
    montage_shots,
    vocal_entries,
)
from tests.test_montage import FPS, make_sync_project, write_wav


@pytest.fixture
def montage(tmp_path):
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    shots = [(0, "A"), (10, "zL"), (20, "L"), (30, "A")]
    build_montage(
        P, shots, {"zL": ("L", 960, 540, 2.0)}, [], tmp_path / "m.mlt", cut_end=40.0
    )
    return tmp_path / "m.mlt"


def test_montage_shots(montage):
    shots = montage_shots(montage)
    assert [s[2] for s in shots] == [
        "A",
        "zL",
        "L",
        "A",
    ]  # the zoom name, else the camera
    assert [round(s[0] * FPS) for s in shots] == [
        0,
        round(10 * FPS),
        round(20 * FPS),
        round(30 * FPS),
    ]
    assert round(shots[-1][1] * FPS) == round(40 * FPS)


def test_vocal_entries(tmp_path):
    sr = 1000
    x = np.zeros(10 * sr)
    x[3 * sr : 4 * sr] = 0.5  # an entry at 3 s ...
    x[4 * sr + 100 : 5 * sr] = 0.5  # ... a gap of 0.1 s is not a new entry
    x[7 * sr : 8 * sr] = 0.5  # a new entry at 7 s
    write_wav(tmp_path / "v.wav", x, sr)
    assert np.allclose(
        vocal_entries(tmp_path / "v.wav", offset=1.0), [2.0, 6.0], atol=0.03
    )


def test_audit_cuts(montage, tmp_path):
    # the reference rush starts at 2 s in the project; the singer sings from 15 to 20 s (project time)
    sr = 1000
    x = np.zeros(60 * sr)
    x[13 * sr : 18 * sr] = 0.5
    write_wav(tmp_path / "vocals.wav", x, sr)
    rows = audit_cuts(
        montage, tmp_path / "vocals.wav", "x_P1000001.MP4", singer_framings={"A", "L"}
    )
    by = {r["framing"]: r for r in rows}
    assert by["zL"]["voice"] == pytest.approx(
        5.0, abs=0.05
    )  # the voice comes 5 s after the cut at 10 s
    assert by["zL"]["sings"] == pytest.approx(0.5, abs=0.05)
    assert by["zL"]["flag"]  # sings half of a shot where he is not framed
    assert not any(r["flag"] for r in rows if r["framing"] != "zL")
