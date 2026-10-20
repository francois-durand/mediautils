"""Audit of a concert montage before it is shown: its cuts against the sound, and its pictures.

Made after the montage of Burlingue's concert (2026-10): a montage computed from the
structure of a song (bars, sections) must be checked against what is really heard
and seen. This module reads a montage project (``.mlt``, e.g. from
:func:`~mediautils.montage.build_montage`) and

- lists its shots, with the name of their framing (:func:`montage_shots`);
- compares every cut with the nearest entry of the singer in the vocals stem, and
  flags the shots where he sings while not framed (:func:`vocal_entries`,
  :func:`audit_cuts`);
- shows three pictures per shot (start, middle, end) on contact sheets, to look for
  cut heads, people hidden or cut by the frame, spectators in front of the camera
  (:func:`audit_frames`).
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mediautils.montage import SyncProject, _clock_of, _prop


def montage_shots(montage: str | os.PathLike) -> list[tuple[float, float, str]]:
    """Shots of a montage project, in seconds: ``(start, end, framing)``.

    The framing is the zoom name written by :func:`~mediautils.montage.build_montage`
    (``montage:shot``), else the camera of the file (``"L"``, ``"A"``, ``"W"``); at each
    moment the shot is the top video track that has a clip there (as in Shotcut).
    ``"black"``: no clip.

    Parameters
    ----------
    montage : str | os.PathLike
        The montage project.

    Returns
    -------
    list[tuple[float, float, str]]
        The shots, in order.
    """
    C, pieces = _visible(montage)
    shots = []
    for a, b, label, *_ in pieces:
        if shots and shots[-1][2] == label and shots[-1][1] == a:
            shots[-1] = (shots[-1][0], b, label)
        else:
            shots.append((a, b, label))
    return [(a / C.fps, b / C.fps, lab) for a, b, lab in shots]


def _visible(montage):
    """What is seen on the timeline: ``(clock, [(a, b, label, chain, in, pos)])`` in frames, where a clip of
    `chain` placed at `pos` with in-point `in` is the top visible clip on [a, b) (``chain=None``: black)."""
    P = SyncProject(montage)
    root = ET.parse(montage).getroot()
    C = _clock_of(root)
    by_id = {el.get("id"): el for el in root}
    main = next(t for t in root if t.tag == "tractor" and _prop(t, "shotcut") == "1")
    tracks = []  # bottom to top: list of (start, end, label, chain, in, pos)
    for tr in main.findall("track"):
        pl = by_id.get(tr.get("producer"))
        if pl is None or pl.tag != "playlist" or _prop(pl, "shotcut:video") != "1":
            continue
        if tr.get("hide") in ("video", "both"):
            continue
        items, pos = [], 0
        for el in pl:
            if el.tag == "blank":
                pos += C.frames(el.get("length"))
            elif el.tag == "entry":
                i0 = C.frames(el.get("in"))
                n = C.frames(el.get("out")) - i0 + 1
                chain = by_id[el.get("producer")]
                label = _prop(chain, "montage:shot") or P.camera(
                    _prop(chain, "resource") or ""
                )
                items.append((pos, pos + n, label, chain, i0, pos))
                pos += n
        tracks.append(items)
    cuts = sorted({x for items in tracks for it in items for x in it[:2]})
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        hit = None
        for items in reversed(tracks):
            hit = next((it for it in items if it[0] <= a < it[1]), None)
            if hit is not None:
                break
        pieces.append((a, b, *hit[2:]) if hit else (a, b, "black", None, 0, 0))
    return C, pieces


def _crop(chain, k, C):
    """ffmpeg crop of the Shotcut "Size, Position & Rotate" filter of `chain` at frame `k` of the source."""
    filt = next(
        (f for f in chain.findall("filter") if _prop(f, "mlt_service") == "affine"),
        None,
    )
    if filt is None:
        return "null"
    k -= (
        C.frames(filt.get("in")) if filt.get("in") else 0
    )  # keyframes count from the filter's in-point
    rect = _prop(filt, "transition.rect")
    if "=" in rect:  # keyframes "f0=x y w h o;f1=..." : linear interpolation
        keys = [
            (int(a), [float(v) for v in r.split()])
            for a, r in (p.split("=") for p in rect.split(";"))
        ]
        (k0, r0), (k1, r1) = keys[0], keys[-1]
        u = min(max((k - k0) / max(k1 - k0, 1), 0), 1)
        x, y, w, h = [a + (b - a) * u for a, b in zip(r0[:4], r1[:4])]
    else:
        x, y, w, h = [float(v) for v in rect.split()[:4]]
    z = w / 1920
    return f"crop={1920 / z:.0f}:{1080 / z:.0f}:{-x / z:.0f}:{-y / z:.0f}"


def vocal_entries(
    vocals: str | os.PathLike,
    offset: float,
    threshold: float = -34.0,
    silence: float = 0.2,
) -> np.ndarray:
    """Entries of the voice: starts of the passages above `threshold` after `silence` s below it.

    Parameters
    ----------
    vocals : str | os.PathLike
        Vocals stem of a rush (e.g. from Demucs).
    offset : float
        Time of the rush = time of the project + `offset` (seconds).
    threshold : float
        Level of the voice (dB, 20 ms windows).
    silence : float
        Minimum silence before an entry (seconds).

    Returns
    -------
    numpy.ndarray
        Times of the entries, in project time.
    """
    import soundfile as sf

    x, sr = sf.read(vocals, always_2d=True)
    x = x.mean(axis=1)
    h = int(sr * 0.02)
    n = len(x) // h
    db = 10 * np.log10(np.mean(x[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12)
    on = db > threshold
    starts, quiet, need = [], 10**9, int(round(silence / 0.02))
    for i in range(n):
        if on[i] and quiet >= need:
            starts.append(i * 0.02 - offset)
        quiet = 0 if on[i] else quiet + 1
    return np.array(starts)


def audit_cuts(
    montage: str | os.PathLike,
    vocals: str | os.PathLike,
    rush: str,
    singer_framings: set[str] | None = None,
    sings_threshold: float = -32.0,
) -> list[dict]:
    """Every cut of a montage against the voice, and the shots where the singer is not framed.

    Parameters
    ----------
    montage : str | os.PathLike
        The montage project.
    vocals : str | os.PathLike
        Vocals stem of `rush`.
    rush : str
        File name of the rush of the stem (its position gives the offset).
    singer_framings : set[str] | None
        Framings where the singer is seen; a shot of another framing where the voice
        is loud more than 30 % of the time is flagged. ``None``: no flag.
    sings_threshold : float
        Level (dB) above which the voice counts as singing.

    Returns
    -------
    list[dict]
        One per shot: ``start``, ``end``, ``framing``, ``voice`` (nearest vocal entry
        minus the cut, s: positive = the voice comes after the cut), ``sings`` (share of
        the shot where the voice is loud), ``flag`` (text, or ``""``).
    """
    import soundfile as sf

    P = SyncProject(montage)
    t0 = dict(P.rushes())[rush]
    offset = -t0 / P.fps
    entries = vocal_entries(vocals, offset)
    x, sr = sf.read(vocals, always_2d=True)
    x = x.mean(axis=1)
    h = int(sr * 0.05)

    def sings(a, b):
        i, j = int((a + offset) * sr), int((b + offset) * sr)
        w = x[max(i, 0) : max(j, 0)]
        w = w[: len(w) // h * h].reshape(-1, h)
        if not len(w):
            return 0.0
        return float(
            np.mean(10 * np.log10(np.mean(w**2, axis=1) + 1e-12) > sings_threshold)
        )

    rows = []
    for a, b, framing in montage_shots(montage):
        near = (
            float(entries[np.argmin(abs(entries - a))] - a)
            if len(entries)
            else float("nan")
        )
        s = sings(a, b)
        flag = ""
        if singer_framings is not None and framing not in singer_framings and s > 0.3:
            flag = f"sings {s:.0%} of the shot, not framed"
        rows.append(
            {
                "start": a,
                "end": b,
                "framing": framing,
                "voice": near,
                "sings": s,
                "flag": flag,
            }
        )
    return rows


def _stack(files, dst, how):
    if len(files) == 1:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(files[0]), str(dst)], check=True
        )
        return
    inp = sum([["-i", str(f)] for f in files], [])
    fc = "".join(f"[{i}]" for i in range(len(files))) + f"{how}=inputs={len(files)}"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inp, "-filter_complex", fc, str(dst)],
        check=True,
    )


def audit_frames(
    montage: str | os.PathLike,
    dst: str | os.PathLike,
    per_sheet: int = 8,
    width: int = 320,
) -> list[tuple[Path, list[tuple[float, float, str]]]]:
    """Three pictures per shot (start + 0.15 s, middle, end - 0.15 s), on sheets.

    Each picture is read from the source file of the visible clip and cropped as its
    zoom (the Shotcut "Size, Position & Rotate" filter, keyframes interpolated), with
    ffmpeg: much faster than rendering the project. Each sheet has one row per shot.
    Look at them for heads cut by the frame, people hidden (behind the singer...) or
    cut at the edge, spectators in front of the camera.

    Parameters
    ----------
    montage : str | os.PathLike
        The montage project.
    dst : str | os.PathLike
        Folder of the pictures and sheets (created).
    per_sheet : int
        Shots per sheet.
    width : int
        Width of a picture.

    Returns
    -------
    list[tuple[Path, list[tuple[float, float, str]]]]
        The sheets, each with its shots.
    """
    montage, dst = Path(montage).resolve(), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    C, pieces = _visible(montage)
    shots = montage_shots(montage)
    height = width * 9 // 16
    fit = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    strips = []
    for k, (a, b, _) in enumerate(shots):
        pics = []
        for j, t in enumerate([a + 0.15, (a + b) / 2, b - 0.15]):
            f = dst / f"shot{k:03d}_{j}.png"
            n = int(round(t * C.fps))
            a_, b_, _, chain, i0, pos = next(
                (p for p in pieces if p[0] <= n < p[1]), pieces[-1]
            )
            if chain is None:
                cmd = ["-f", "lavfi", "-i", f"color=black:s={width}x{height}"]
                vf = "null"
            else:
                src = montage.parent / _prop(chain, "resource")
                k_src = n - pos + i0  # frame of the source
                cmd = ["-ss", f"{k_src / C.fps:.3f}", "-i", str(src)]
                vf = f"{_crop(chain, k_src, C)},{fit}"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    *cmd,
                    "-frames:v",
                    "1",
                    "-vf",
                    vf,
                    str(f),
                ],
                check=True,
            )
            pics.append(f)
        strip = dst / f"shot{k:03d}.png"
        inp = sum([["-i", str(f)] for f in pics], [])
        fc = "[0][1][2]hstack=inputs=3"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *inp, "-filter_complex", fc, str(strip)],
            check=True,
        )
        strips.append(strip)
    sheets = []
    for s in range(0, len(strips), per_sheet):
        sheet = dst / f"sheet{s // per_sheet:02d}.jpg"
        _stack(strips[s : s + per_sheet], sheet, "vstack")
        sheets.append((sheet, shots[s : s + per_sheet]))
    return sheets
