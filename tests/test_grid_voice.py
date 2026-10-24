import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mediautils import (
    BarGrid,
    Voice,
    bar_markers,
    fill_beats,
    numbered_beat_markers,
    snap,
    subdivide,
)
from tests.test_montage import make_sync_project

sf = pytest.importorskip("soundfile")


def markers(path):
    main = ET.parse(path).getroot().find("tractor[@id='tractor1']")
    out = []
    for m in main.findall("properties[@name='shotcut:markers']/properties"):
        out.append(
            (
                m.find("property[@name='text']").text,
                m.find("property[@name='color']").text,
            )
        )
    return out


def test_grid_pipeline():
    # detector on the 2 and 4 (one beat out of two), a missed beat where the drums stop
    quarters = np.arange(0, 40) * 0.5 + 0.25
    detected = np.delete(quarters[1::2], 9)
    beats = fill_beats(subdivide(detected))
    assert len(beats) == len(quarters) - 1
    assert np.allclose(beats, quarters[1:])
    g = BarGrid.from_downbeat(beats, 2.25, bar=2)  # the user: "2.25 is the 1 of bar 2"
    assert g.label(0) == (1, 2)
    assert g.time(3) == pytest.approx(4.25)
    assert g.time(3, 4.5) == pytest.approx(6.0)
    assert g.is_downbeat(8.27) and not g.is_downbeat(8.75)
    assert snap([4.25], [4.29])[0] == pytest.approx(4.29)


def test_from_downbeat_shift_and_snap():
    # the detector lags the attacks by 0.2 s; the drums hit 0.03 s after the true beats
    true = np.arange(0, 32) * 0.5
    g = BarGrid.from_downbeat(true + 0.2, 4.0, bar=3, onsets=true + 0.03)
    assert g.time(3) == pytest.approx(4.03)
    assert g.time(1) == pytest.approx(0.03)


def test_between_drift_and_jitter():
    # the tempo slows down slowly; one beat out of two detected, a few of them off
    rng = np.random.default_rng(1)
    true = np.cumsum(np.linspace(0.5, 0.51, 200))
    halves = true[::2] + np.where(
        rng.random(100) < 0.15, rng.choice([-0.03, 0.03], 100), 0
    )
    detected = subdivide(halves)
    t1, t2 = true[20], true[180]  # 1s of bars 6 and 46
    g = BarGrid.between(detected, (t1, 6), (t2, 46))
    assert g.time(6) == pytest.approx(t1) and g.time(46) == pytest.approx(t2)
    err = [g.time(m) - true[20 + 4 * (m - 6)] for m in range(6, 47)]
    assert np.max(np.abs(err)) < 0.04
    raw = BarGrid.between(detected, (t1, 6), (t2, 46), smooth=False)
    assert raw.time(26) == pytest.approx(true[100], abs=0.06)
    with pytest.raises(ValueError):
        BarGrid.between(detected, (t1, 6), (t2, 50))  # 16 beats too many


def test_markers(tmp_path):
    p = make_sync_project(tmp_path / "p.mlt")
    beats = np.arange(1, 11) * 0.5
    numbered_beat_markers(p, beats)
    assert [m[0] for m in markers(p)][:3] == ["1", "2", "3"]
    q = make_sync_project(tmp_path / "q.mlt")
    g = BarGrid(beats, index=2, bar=1)  # beats[0] is beat 3 of bar 0: skipped
    bar_markers(q, g, sections={1: "Intro", 2: "Couplet 1"}, until=4.6)
    texts = dict(markers(q))
    assert texts["1.1 Intro"] == "#30d050"
    assert texts["2.1 Couplet 1"] == "#30d050"
    assert texts["1.2"] == "#5a6a80"
    assert "3.1" not in texts  # after `until`
    r = make_sync_project(tmp_path / "r.mlt")
    bar_markers(
        r, BarGrid(beats, index=0, bar=2), sections={1: "Intro"}
    )  # bar 1 cut: section at 0:00
    assert ("Intro", "#30d050") in markers(r)


def test_voice(tmp_path):
    sr = 8000
    t = np.arange(int(sr * 10)) / sr
    voice = np.zeros_like(t)
    for a, b in (
        (1.2, 3.0),
        (3.15, 4.0),
        (6.3, 8.0),
    ):  # phrase, breath of 0.15 s, phrase ; then a pause
        voice[(t >= a) & (t < b)] = 0.3 * np.sin(
            2 * np.pi * 220 * t[(t >= a) & (t < b)]
        )
    path = tmp_path / "vocals.wav"
    sf.write(path, np.stack([voice, voice], axis=1), sr)
    v = Voice(path, offset=0.0)
    beats = np.arange(0, 10, 0.5)
    assert v.entry(5, 7) == pytest.approx(6.3, abs=0.05)
    assert v.breath_cut(6.3, beats) == pytest.approx(
        6.0
    )  # the beat before, silent: his breath
    assert v.line(5, 7, beats, default=6.5) == pytest.approx(6.0)
    assert v.line(9, 9.9, beats, default=9.5) == 9.5
    assert v.phrase_end(1.5, beats) == pytest.approx(
        4.0, abs=0.06
    )  # the short breath does not end it
    assert v.sings(1.2, 3.0) > 0.9 and v.sings(4.5, 6.0) == 0
    assert v.longest_pause(4.0, 6.3) == pytest.approx(5.15, abs=0.1)


def test_estimate_shifts():
    cv2 = pytest.importorskip("cv2")
    from mediautils.stabilize import combine_shifts, estimate_shifts

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(
        rng.uniform(0, 255, (120, 400)).astype(np.float32), (0, 0), 2
    )
    true = [(0, 0), (1.5, -2.0), (-1.0, 3.0), (0, 0)]
    frames = [
        cv2.warpAffine(
            base,
            np.float32([[1, 0, dx], [0, 1, dy]]),
            (400, 120),
            borderMode=cv2.BORDER_REFLECT,
        )
        for dx, dy in true
    ]
    s = estimate_shifts(frames, [(20, 180), (220, 380)], reference=base)
    c, bad = combine_shifts(s)
    assert bad == 0
    assert np.allclose(c - c[0], np.array(true) - true[0], atol=0.3)
