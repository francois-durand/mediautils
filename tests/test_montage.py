import shutil
import wave

import numpy as np
import pytest

from mediautils import (
    MltClock,
    SyncProject,
    build_montage,
    nervous_shots,
    resync_project,
    trim_project_end,
    trim_project_head,
    use_single_audio,
)

FPS = 60000 / 1001
C = MltClock(FPS)


def make_sync_project(
    path,
    ref="x_P1000001.MP4",
    other="y_MVI_0001.MOV",
    ref_start=2.0,
    ref_len=60.0,
    other_t0=0,
    other_len=55.0,
    fade=3.0,
):
    """A synchronization project as Shotcut writes it: V1 = reference camera (starting at `ref_start`),
    V2 = the other camera, an audio track going from the other camera to the reference with a cross-fade."""
    rs, rl, ol, fd = (
        round(ref_start * FPS),
        round(ref_len * FPS),
        round(other_len * FPS),
        round(fade * FPS),
    )
    end = rs + rl

    def chain(cid, res, n, video):
        return (
            f'<chain id="{cid}" out="{C.string(n - 1)}"><property name="length">{C.string(n)}</property>'
            f'<property name="resource">{res}</property>'
            f'<property name="mlt_service">avformat-novalidate</property>'
            f'<property name="audio_index">{-1 if video else 1}</property>'
            f'<property name="video_index">{0 if video else -1}</property>'
            f'<property name="shotcut:caption">{res}</property></chain>'
        )

    xml = f"""<?xml version="1.0" standalone="no"?>
<mlt LC_NUMERIC="C" version="7.5.0" producer="main_bin">
  <profile description="automatic" width="1920" height="1080" progressive="1" frame_rate_num="60000" frame_rate_den="1001"/>
  <playlist id="main_bin"><property name="xml_retain">1</property></playlist>
  <producer id="black" in="{C.string(0)}" out="{C.string(end - 1)}"><property name="length">{C.string(end)}</property>
    <property name="resource">0</property><property name="mlt_service">color</property></producer>
  <playlist id="background"><entry producer="black" in="{C.string(0)}" out="{C.string(end - 1)}"/></playlist>
  {chain("chain0", ref, rl, True)}
  <playlist id="playlist0"><property name="shotcut:video">1</property><property name="shotcut:name">V1</property>
    <blank length="{C.string(rs)}"/><entry producer="chain0" in="{C.string(0)}" out="{C.string(rl - 1)}"/></playlist>
  {chain("chain1", other, ol, True)}
  <playlist id="playlist1"><property name="shotcut:video">1</property><property name="shotcut:name">V2</property>
    <entry producer="chain1" in="{C.string(-other_t0)}" out="{C.string(ol - 1)}"/></playlist>
  {chain("chain2", other, ol, False)}
  {chain("chain3", ref, rl, False)}
  <tractor id="tractor0" in="{C.string(0)}" out="{C.string(fd - 1)}">
    <track producer="chain2" in="{C.string(rs - other_t0)}" out="{C.string(rs - other_t0 + fd - 1)}"/>
    <track producer="chain3" in="{C.string(0)}" out="{C.string(fd - 1)}"/>
    <transition id="transition0" out="{C.string(fd - 1)}"><property name="a_track">0</property>
      <property name="b_track">1</property><property name="mlt_service">mix</property></transition>
  </tractor>
  <playlist id="playlist2"><property name="shotcut:audio">1</property><property name="shotcut:name">Son droite</property>
    <entry producer="chain2" in="{C.string(-other_t0)}" out="{C.string(rs - other_t0 - 1)}"/>
    <entry producer="tractor0" in="{C.string(0)}" out="{C.string(fd - 1)}"/>
    <entry producer="chain3" in="{C.string(fd)}" out="{C.string(rl - 1)}"/>
    <filter id="filter0"><property name="mlt_service">panner</property></filter></playlist>
  <playlist id="playlist3"><property name="shotcut:audio">1</property><property name="shotcut:name">Son gauche</property>
    <entry producer="chain2" in="{C.string(-other_t0)}" out="{C.string(ol - 1)}"/></playlist>
  <tractor id="tractor1" in="{C.string(0)}" out="{C.string(end - 1)}"><property name="shotcut">1</property>
    <track producer="background"/><track producer="playlist0"/><track producer="playlist1"/>
    <track producer="playlist2" hide="video"/><track producer="playlist3" hide="video"/></tractor>
</mlt>
"""
    path.write_text(xml, encoding="utf-8")
    return path


def write_wav(path, x, sr):
    x = np.atleast_2d(np.asarray(x).T).T if np.ndim(x) == 1 else np.asarray(x)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_sync_project(tmp_path):
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    assert P.segments["L"] == [(120, 120 + round(60 * FPS), "chain0", 120)]
    assert P.segments["A"] == [(0, round(55 * FPS), "chain1", 0)]
    assert [name for name, _ in P.rushes()] == ["y_MVI_0001.MOV", "x_P1000001.MP4"]
    ((a, b, files),) = P.crossfades()
    assert (round(a, 2), round(b, 2), files) == (
        2.0,
        5.0,
        {"y_MVI_0001.MOV", "x_P1000001.MP4"},
    )
    assert P.covering("L", 0, 10) is None and P.covering("A", 0, 10) == ("chain1", 0)


def test_build_montage_and_audio(tmp_path):
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    zooms = {"zL": ("L", 960, 540, 2.0), "zA": ("A", 960, 540, 2.0)}
    shots = nervous_shots([(0, 20, ["L", "zA"]), (20, 50, ["zL", "A", "L"])])
    dst = tmp_path / "montage.mlt"
    segs = build_montage(P, shots, zooms, [], dst, cut_end=50.0, trim_audio=False)
    # the first shot asks for the reference camera before it films: the other camera instead
    assert segs[0][2] == "A" and segs[0][0] == 0
    assert all(b - a >= 2.5 * FPS for a, b, _, _ in segs)
    M = SyncProject(dst)
    assert len(M.segments["A"]) >= 2 and any(
        cid.startswith("montage_zoom") for _, _, cid, _ in M.segments["L"]
    )
    assert max(e for segs_ in M.segments.values() for _, e, _, _ in segs_) == round(
        50 * FPS
    )
    with pytest.raises(FileExistsError):
        build_montage(P, shots, zooms, [], dst)
    # single audio file, head cut
    write_wav(
        tmp_path / "mix.wav", np.zeros((50 * 1000, 2)), 1000
    )  # as long as the montage
    final = tmp_path / "final.mlt"
    use_single_audio(dst, final, "mix.wav")
    text = final.read_text(encoding="utf-8")
    assert (
        "tractor0" not in text
        and "panner" not in text
        and 'producer="single_audio"' in text
    )
    trim_project_head(final, 3.0)
    F = SyncProject(final)
    assert F.end == round(50 * FPS) - round(3 * FPS)
    assert 'producer="single_audio" in="00:00:03.003"' in final.read_text(
        encoding="utf-8"
    )
    trim_project_end(final, 40.0)
    assert SyncProject(final).end == round(40 * FPS)


def test_trim_end_refuses_cross_fade(tmp_path):
    p = make_sync_project(tmp_path / "p.mlt")
    with pytest.raises(ValueError):
        trim_project_end(p, 3.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed")
def test_resync_project(tmp_path):
    sr = 16000
    rng = np.random.default_rng(1)
    event = rng.standard_normal(40 * sr) * 0.2
    write_wav(
        tmp_path / "x_P1000001.wav", event[5 * sr : 35 * sr], sr
    )  # starts 5 s after the other one
    write_wav(tmp_path / "y_MVI_0001.wav", event[: 30 * sr], sr)
    # the other camera placed 2 frames too early
    src = make_sync_project(
        tmp_path / "p.mlt",
        ref="x_P1000001.wav",
        other="y_MVI_0001.wav",
        ref_start=5.0,
        ref_len=30.0,
        other_t0=-2,
        other_len=30.0,
    )
    report = resync_project(src, tmp_path / "q.mlt", reference=r"_P\d{7}\.wav$")
    assert len(report) == 2
    Q = SyncProject(tmp_path / "q.mlt", reference=r"_P\d{7}\.wav$")
    assert Q.segments["A"][0][3] == round(5 * FPS) - round(5 * FPS)  # t0 = 0
    assert {t0 for _, _, _, t0 in Q.segments["L"]} == {round(5 * FPS)}
