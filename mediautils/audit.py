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
    r = _crop_rect(chain, k, C)
    return "null" if r is None else "crop={2:.0f}:{3:.0f}:{0:.0f}:{1:.0f}".format(*r)


def _crop_rect(chain, k, C):
    """Part of the 1920x1080 source shown by `chain` at its frame `k`: ``(x0, y0, w, h)``, or ``None`` (all)."""
    filt = next(
        (f for f in chain.findall("filter") if _prop(f, "mlt_service") == "affine"),
        None,
    )
    if filt is None:
        return None
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
    return (-x / z, -y / z, 1920 / z, 1080 / z)


def audit_headroom(
    montage: str | os.PathLike,
    step: float = 0.5,
    top_margin: float = 0.02,
    edge: float = 0.03,
    around: float = 2.0,
    subjects: dict[str, list[float]] | None = None,
    reach: float = 220.0,
) -> list[dict]:
    """Heads cut by the frame, and people cut at its edge, shot by shot, from the faces in the source.

    Faces are looked for every `step` s in the whole source picture (OpenCV Haar
    cascades, frontal and profile), so that a head the zoom cuts is still found, from
    `around` s before to `around` s after the shot (its zoom is fixed: a face seen just
    before the shot tells where the frame stands when the person looks down or away
    during it). For a face whose center is in the zoom, the top of the head is taken a
    third of a face above the face; the margin is its distance to the top edge of the
    zoom, as a fraction of the zoom's height (negative: the head is cut).

    A face looking down all along, or hidden by the microphone, may be missed: this
    complements the sheets of :func:`audit_frames`, it does not replace looking at them.

    Parameters
    ----------
    montage : str | os.PathLike
        The montage project.
    step : float
        Time between two checks (seconds).
    top_margin : float
        Minimum margin above the heads.
    edge : float
        A face whose center is within this fraction of the zoom's width from its left or
        right edge is "cut at the edge".
    around : float
        Seconds checked before and after each shot.
    subjects : dict[str, list[float]] | None
        Framing -> x of the people it is about (as given to :func:`fit_zooms`): only their
        heads count (faces within `reach` px). Without it, the people in the middle 70 %
        of the zoom.
    reach : float
        See `subjects`.

    Returns
    -------
    list[dict]
        One per shot with a problem: ``start``, ``end`` (s), ``framing``, ``margin`` (the
        smallest margin above a head, or ``None``), ``issues`` (``"head"``, ``"edge"``).
    """
    import cv2

    montage = Path(montage).resolve()
    C, pieces = _visible(montage)
    cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + name)
        for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml")
    ]
    shots = []  # pieces of a same clip glued back
    for p in pieces:
        if shots and shots[-1][3] is p[3] and shots[-1][1] == p[0]:
            shots[-1] = (shots[-1][0], p[1], *shots[-1][2:])
        else:
            shots.append(p)
    caps, rows = {}, []
    stride = max(1, int(round(step * C.fps)))
    pad = int(round(around * C.fps))
    for a, b, label, chain, i0, pos in shots:
        if chain is None:
            continue
        src = str(montage.parent / _prop(chain, "resource"))
        cap = caps.setdefault(src, cv2.VideoCapture(src))
        inside, outside, issues = [], [], set()
        for n in range(max(a - pad, pos), b + pad, stride):
            k_src = n - pos + i0
            cap.set(cv2.CAP_PROP_POS_FRAMES, k_src)
            ok, img = cap.read()
            if not ok:
                continue
            H, W = img.shape[:2]
            gray = cv2.cvtColor(
                cv2.resize(img, (960, int(960 * H / W))), cv2.COLOR_BGR2GRAY
            )
            f = W / 960  # detection picture -> source pixels
            faces = []
            for c in cascades:
                faces += [
                    tuple(v) for v in c.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
                ]
                if c is cascades[1]:  # profiles looking the other way
                    flipped = c.detectMultiScale(
                        cv2.flip(gray, 1), 1.1, 5, minSize=(24, 24)
                    )
                    faces += [(960 - x - w, y, w, h) for x, y, w, h in flipped]
            # the zoom of the shot (fixed zooms; for a progressive one, its state at the nearest frame)
            k_rect = min(max(n, a), b - 1) - pos + i0
            x0, y0, cw, ch = _crop_rect(chain, k_rect, C) or (0, 0, W, H)
            for x, y, w, h in faces:
                x, y, w, h = x * f, y * f, w * f, h * f
                cx = x + w / 2
                # only the people the shot is about (its subjects, else the middle 70 % of the zoom): a
                # close-up on someone may cut the neighbours (accepted by Fanfan); a face wholly above
                # the zoom is not in it
                xs = (subjects or {}).get(label)
                about = (
                    any(abs(cx - s) < reach for s in xs)
                    if xs
                    else abs(cx - (x0 + cw / 2)) <= 0.35 * cw
                )
                if not about or not x0 <= cx <= x0 + cw or y + h < y0:
                    continue
                m = (y - h / 3 - y0) / ch
                (inside if a <= n < b else outside).append(m)
                if a <= n < b and (cx - x0 < edge * cw or x0 + cw - cx < edge * cw):
                    issues.add("edge")
        # the faces seen during the shot; those around it only if none was seen during it (a person
        # looking down all along): around the shot, people stand up straighter than in it
        seen = inside or outside
        margin = min(seen) if seen else None
        if margin is not None and margin < top_margin:
            issues.add("head")
        if issues:
            rows.append(
                {
                    "start": a / C.fps,
                    "end": b / C.fps,
                    "framing": label,
                    "margin": margin,
                    "issues": sorted(issues),
                }
            )
    for cap in caps.values():
        cap.release()
    return rows


def fit_zooms(
    source: str | os.PathLike,
    framings: dict[str, tuple[str, list[float], float]],
    start: float = 0.0,
    end: float | None = None,
    step: float = 1.0,
    target: float = 0.08,
    percentile: float = 2.0,
    reach: float = 220.0,
    min_margin: float = 0.03,
) -> dict[str, tuple[str, float, float, float]]:
    """Zooms of a song fitted to where its people stand: ``name -> (camera, centre_x, centre_y, factor)``.

    Each framing is given by the people it shows, ``(camera, [x of each person], factor)``
    (approximate x in the 1920-wide picture of `source`, the file of `camera`). The faces
    detected every `step` s within `reach` px of each given x tell where the person really
    stands: the zoom is centred on the median positions of its people, its height puts
    the median top of their heads `target` of the zoom's height under its top edge,
    unless that would leave less than `min_margin` above more than `percentile` % of
    their heads. Other people may be
    cut by a zoom (accepted: a close-up on someone cuts the neighbours).

    The framing of a camera changes from song to song: fit the zooms of each song.
    """
    import cv2

    cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + name)
        for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml")
    ]
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if end is None:
        end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    faces = []
    for t in np.arange(start, end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok:
            continue
        f = 1920 / img.shape[1]  # to the 1920-wide coordinates of the zooms
        g = img.shape[1] / 960
        gray = cv2.cvtColor(
            cv2.resize(img, (960, int(img.shape[0] / g))), cv2.COLOR_BGR2GRAY
        )
        for c in cascades:
            for x, y, w, h in c.detectMultiScale(gray, 1.1, 5, minSize=(24, 24)):
                faces.append(((x + w / 2) * g * f, (y - h / 3) * g * f))
    cap.release()
    faces = np.array(faces).reshape(-1, 2)
    out = {}
    for name, (cam, xs, z) in framings.items():
        w, h = 1920 / z, 1080 / z
        people = [faces[abs(faces[:, 0] - x) < reach] for x in xs]
        if any(len(p) == 0 for p in people):
            continue
        cx = float(np.mean([np.median(p[:, 0]) for p in people]))
        best = min(np.median(p[:, 1]) for p in people) - target * h
        safe = min(np.percentile(p[:, 1], percentile) for p in people) - min_margin * h
        y0 = min(best, safe)
        cy = float(min(max(y0 + h / 2, h / 2), 1080 - h / 2))
        out[name] = (cam, float(min(max(cx, w / 2), 1920 - w / 2)), cy, z)
    return out


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
