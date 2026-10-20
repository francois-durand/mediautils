import shutil
import wave
import xml.etree.ElementTree as ET

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
    phone=None,
):
    """A synchronization project as Shotcut writes it: V1 = reference camera (starting at `ref_start`),
    V2 = the other camera, an audio track going from the other camera to the reference with a cross-fade;
    `phone` = (start_s, length_s): a WhatsApp video on V3."""
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

    phone_xml = phone_track = ""
    if phone:
        ps, pn = round(phone[0] * FPS), round(phone[1] * FPS)
        phone_xml = (
            chain("chainW", "z-WA0001.mp4", pn, True)
            + '<playlist id="playlistW"><property name="shotcut:video">1</property>'
            + '<property name="shotcut:name">V3</property>'
            + f'<blank length="{C.string(ps)}"/>'
            + f'<entry producer="chainW" in="{C.string(0)}" out="{C.string(pn - 1)}"/></playlist>'
        )
        phone_track = '<track producer="playlistW"/>'
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
  {phone_xml}
  <tractor id="tractor1" in="{C.string(0)}" out="{C.string(end - 1)}"><property name="shotcut">1</property>
    <track producer="background"/><track producer="playlist0"/><track producer="playlist1"/>{phone_track}
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


def test_build_montage_blocked(tmp_path):
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    zooms = {"zL": ("L", 960, 540, 2.0)}
    shots = nervous_shots([(0, 62, ["zL", "A"])])
    segs = build_montage(
        P,
        shots,
        zooms,
        [],
        tmp_path / "m.mlt",
        blocked={"L": [(20, 30)], "A": [(40, None)]},
    )
    for a, b, cam, _ in segs:
        if cam == "L":  # someone in front of L from 20 to 30 s
            assert b <= round(20 * FPS) or round(30 * FPS) <= a
        else:  # and in front of A from 40 s to the end
            assert b <= round(40 * FPS)
    assert {cam for _, _, cam, _ in segs} == {"A", "L"}
    # nobody may be shown from 50 s if L is blocked too: left black
    segs = build_montage(
        P,
        shots,
        zooms,
        [],
        tmp_path / "n.mlt",
        blocked={"L": [(50, None)], "A": [(40, None)]},
    )
    assert segs[-1][2] is None and segs[-1][0] == round(50 * FPS)
    M = SyncProject(tmp_path / "n.mlt")
    assert all(e <= round(50 * FPS) for v in M.segments.values() for _, e, _, _ in v)


def test_build_montage_blocked_keeps_other_shots(tmp_path):
    # a block on L starting inside a shot of A must not cut that shot into pieces that get absorbed
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    zooms = {"zA": ("A", 960, 540, 2.0), "zL": ("L", 960, 540, 2.0)}
    shots = [(0, "A"), (10, "zL"), (20, "zA"), (24, "L")]
    segs = build_montage(
        P, shots, zooms, [], tmp_path / "m.mlt", blocked={"L": [(21.0, 22.5)]}
    )
    assert [round(20 * FPS), round(24 * FPS), "A", "zA"] in segs


def test_single_audio_not_longer_than_project(tmp_path):
    # a sound longer than the pictures would show a white frame at the end
    P = SyncProject(make_sync_project(tmp_path / "p.mlt"))
    m = tmp_path / "m.mlt"
    build_montage(P, [(0, "A")], {}, [], m, cut_end=40.0, trim_audio=False)
    write_wav(tmp_path / "long.wav", np.zeros((45 * 1000, 2)), 1000)
    use_single_audio(m, tmp_path / "f.mlt", "long.wav")
    assert SyncProject(tmp_path / "f.mlt").end == round(40 * FPS)


SINGLE_CAMERA = """<?xml version="1.0" standalone="no"?>
<mlt LC_NUMERIC="C" version="7.5.0" producer="main_bin">
  <profile width="1920" height="1080" frame_rate_num="60000" frame_rate_den="1001"/>
  <playlist id="main_bin"><property name="xml_retain">1</property></playlist>
  <producer id="black" in="00:00:00.000" out="00:00:09.993">
    <property name="length">00:00:10.010</property><property name="resource">0</property>
    <property name="mlt_service">color</property>
  </producer>
  <playlist id="background"><entry producer="black" in="00:00:00.000" out="00:00:09.993"/></playlist>
  <chain id="chain0" out="00:00:19.987">
    <property name="length">00:00:20.003</property><property name="resource">x_P1000001.MP4</property>
    <property name="mlt_service">avformat-novalidate</property>
    <property name="audio_index">1</property><property name="video_index">0</property>
  </chain>
  <playlist id="playlist0">
    <property name="shotcut:video">1</property><property name="shotcut:name">V1</property>
    <entry producer="chain0" in="00:00:01.001" out="00:00:10.994"/>
  </playlist>
  <tractor id="tractor0" in="00:00:00.000" out="00:00:09.993">
    <property name="shotcut">1</property>
    <track producer="background"/><track producer="playlist0"/>
  </tractor>
</mlt>
"""


def test_single_audio_single_camera(tmp_path):
    # a project whose only sound is that of the (single) camera on its video track
    (tmp_path / "p.mlt").write_text(SINGLE_CAMERA, encoding="utf-8")
    write_wav(tmp_path / "s.wav", np.zeros((12 * 1000, 2)), 1000)
    use_single_audio(tmp_path / "p.mlt", tmp_path / "f.mlt", "s.wav")
    root = ET.parse(tmp_path / "f.mlt").getroot()
    main = root.find("tractor[@id='tractor0']")
    hide = {t.get("producer"): t.get("hide") for t in main.findall("track")}
    assert hide == {"background": None, "playlist0": "audio", "playlist_audio": "video"}
    entry = root.find("playlist[@id='playlist_audio']/entry")
    assert entry.get("producer") == "single_audio"
    assert (
        root.find("chain[@id='single_audio']/property[@name='video_index']").text
        == "-1"
    )
    assert main.find("transition/property[@name='b_track']").text == "2"


def test_build_montage_phone(tmp_path):
    # a third source (WhatsApp video, "W") from 20 to 35 s, on its own track
    P = SyncProject(make_sync_project(tmp_path / "p.mlt", phone=(20.0, 15.0)))
    assert P.segments["W"] == [
        (round(20 * FPS), round(35 * FPS), "chainW", round(20 * FPS))
    ]
    shots = [(0, "A"), (10, "W"), (24, "W"), (30, "L"), (40, "W")]
    segs = build_montage(P, shots, {}, [], tmp_path / "m.mlt")
    f = round
    assert [
        f(10 * FPS),
        f(20 * FPS),
        "L",
        None,
    ] in segs  # the phone does not film yet: the reference
    assert [f(20 * FPS), f(30 * FPS), "W", None] in segs
    assert segs[-1][2] == "L"  # after the phone stops (35 s): the reference
    M = SyncProject(tmp_path / "m.mlt")
    assert [(s, e) for s, e, _, _ in M.segments["W"]] == [(f(20 * FPS), f(30 * FPS))]
    # no reference under the (vertical) phone video: black bars rather than the other camera around it
    assert all(e <= f(20 * FPS) or s >= f(30 * FPS) for s, e, _, _ in M.segments["L"])


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
    # the reference placed 3 frames later: the other camera, whose track starts at its first frame,
    # must start 3 frames after the head of its track -> a blank is inserted before it
    src2 = make_sync_project(
        tmp_path / "p2.mlt",
        ref="x_P1000001.wav",
        other="y_MVI_0001.wav",
        ref_start=5.0 + 3 / FPS,
        ref_len=30.0,
        other_len=30.0,
    )
    resync_project(src2, tmp_path / "q2.mlt", reference=r"_P\d{7}\.wav$")
    Q2 = SyncProject(tmp_path / "q2.mlt", reference=r"_P\d{7}\.wav$")
    ((s, e, _, t0),) = Q2.segments["A"]
    assert (
        t0 == s > 0
    )  # a blank before the other camera, which starts at its first frame


def test_add_markers(tmp_path):
    from mediautils import add_markers

    p = make_sync_project(tmp_path / "p.mlt")
    add_markers(p, [(0.0, "Intro"), (10.0, "Couplet 1")])
    add_markers(p, [(20.0, "Refrain")])  # the markers already there are kept
    main = ET.parse(p).getroot().find("tractor[@id='tractor1']")
    marks = main.findall("properties[@name='shotcut:markers']/properties")
    assert [m.find("property[@name='text']").text for m in marks] == [
        "Intro",
        "Couplet 1",
        "Refrain",
    ]
    start = marks[1].find("property[@name='start']").text
    assert C.frames(start) == round(10 * FPS)
