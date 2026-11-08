"""Checks of a montage before it is shown: alarms, no automatism.

Rules agreed with the user (2026-11-08, ``Règles de montage.md``, section 9):

- **the code flags, Claude looks and decides**: no alarm changes a shot (automatic
  rules had degraded the montages of November 2026: shots recomputed one by one, a shot
  silently replaced by the wide shot, frames moved for the heads...);
- **an alarm is not let through without looking at the picture**; looking does not mean
  correcting: Claude may keep the shot, but he decides it, and notes in one line what he
  did and why (:func:`alarm_log`);
- **the figures are starting points** (:class:`Thresholds`), calibrated lightly: a shot the
  user retouches while no alarm sounded shows an alarm too permissive; an alarm Claude
  nearly always lets through, without the user minding, is too alarmist; a quick review
  at the validation of each song, thresholds changed only on a clear trend.

The alarms (:func:`shot_alarms`), on the pictures of each shot (named keypoints of
:func:`mediautils.identify.identify`):

- **heads**: a head too close to the top edge, or too much air above the heads;
- **margins**: empty space on one side of a group; a person alone off-centre; a wide
  shot with too large a margin;
- **people cut**: someone cut in two by an edge, or a body without its head in the frame;
- **the singer**: singing near an edge and turned towards it; not in the picture while
  singing;
- **musicians**: someone of the shot hidden or not found; someone not identified in the
  frame (a spectator?); a musician who moved away from his place;
- **instruments**: the hands of a guitarist out of the frame; fixed things (toms, ride)
  cut by the frame;
- **chaining**: two framings too close in a row (same people and a zoom ratio under 1.4,
  or overlapping by more than half); going back to a wider framing of the same people;
- **cuts**: a cut in the middle of a phrase of the singer without a breath nor an accent
  to justify it; a cut far from any attack.

The margins are judged on the whole picture with the frame drawn, not only on the
cropped picture (a check of the people cut only had let through a wide shot too empty on
the right, *C'est comme ça*, 2026-11-05): :func:`draw_frame`.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mediautils.catalogue import (
    TIGHT,
    VERY_TIGHT,
    WIDE,
    Extent,
    base_name,
    framing_rect,
    intended,
)
from mediautils.framing import Person
from mediautils.identify import who
from mediautils.pose import (
    WRISTS,
    body_box,
    eyes_y,
    face_direction,
    head_scale,
    head_top,
    head_x,
    seen,
    shoulders_x,
)

W, H = 1920, 1080


@dataclass
class Thresholds:
    """Starting points of the alarms, calibrated by Claude (shares of the frame unless
    stated).

    Attributes
    ----------
    head_top : float
        A head closer than this to the top edge (share of the frame height).
    air : float
        More air than this above the highest head (share of the frame height).
    group_side : float
        Empty space on one side of a group.
    alone_offset : float
        A person alone off-centre by more than this.
    wide_side : float
        Margin of the wide shot on a side.
    cut : tuple[float, float]
        An edge falls on someone, who keeps between these shares of their width in the
        frame.
    cut_alone : tuple[float, float]
        The same for a neighbour in a shot of one person: a small piece of them is
        accepted (rule 3.1, « tant pis »); from a third, the intention of the shot gets
        ambiguous (« le cul entre deux chaises »). Calibrated on *People have the power*
        (2026-11-08), where the shots on Fanfan alone always showed a bit of Thomas.
    singer_edge : float
        The singer's head closer than this to a side edge.
    gaze : float
        Face direction (:func:`~mediautils.pose.face_direction`) beyond which a face is
        turned to a side.
    sings : float
        Share of a shot where the singer sings, for him to be expected in the picture.
    share : float
        Share of the pictures of a shot where a problem must hold to sound.
    moved : float
        Shift of a person from their usual place.
    zoom_ratio : float
        Two framings of the same people in a row need at least this zoom ratio.
    iou : float
        Two framings in a row overlapping more than this (intersection over union).
    phrase : float
        Half-window around a cut where a continuous voice means "the middle of a
        phrase" (seconds).
    accent : float
        An accent within this of a cut justifies it (seconds).
    attack : float
        A cut further than this from any attack (seconds).
    """

    head_top: float = 0.06
    air: float = 0.18
    group_side: float = 0.12
    alone_offset: float = 0.12
    wide_side: float = 0.06
    cut: tuple[float, float] = (0.15, 0.85)
    cut_alone: tuple[float, float] = (0.35, 0.85)
    singer_edge: float = 0.04
    gaze: float = 0.3
    sings: float = 0.3
    share: float = 0.3
    moved: float = 0.08
    zoom_ratio: float = 1.4
    iou: float = 0.5
    phrase: float = 0.15
    accent: float = 0.08
    attack: float = 0.15


@dataclass
class Alarm:
    """An alarm on a shot (or on a cut, `start` = `end`)."""

    start: float
    end: float
    shot: str
    kind: str
    message: str
    pictures: list[float] = field(default_factory=list)


def clock(t: float) -> str:
    """Time of a video as the user reads it, ``m:ss,s``.

    >>> clock(140.84)
    '2:20,8'
    """
    m, s = divmod(round(t, 1), 60)
    return f"{int(m)}:{s:04.1f}".replace(".", ",")


def plan(shots: list[tuple[float, str]], end: float) -> list[tuple[float, float, str]]:
    """Shots ``(start, name)`` as ``(start, end, name)``, the last one ending at `end`.

    >>> plan([(0, "large"), (5.5, "Fanfan")], 12)
    [(0, 5.5, 'large'), (5.5, 12, 'Fanfan')]
    """
    s = sorted(shots)
    return [
        (t, s[k + 1][0] if k + 1 < len(s) else end, n) for k, (t, n) in enumerate(s)
    ]


def _boxes(frame, extend):
    """Named bodies of a picture: name -> (x0, top, x1, bottom, keypoints)."""
    out = {}
    for p in frame["people"]:
        name = p.get("name")
        box = body_box(p["kp"])
        if name is None or box is None:
            continue
        e = extend.get(name, Extent())
        out[name] = (
            box[0] - e.left,
            box[1],
            box[2] + e.right,
            box[3] + e.below,
            p["kp"],
        )
    return out


def _inside(box, rect):
    """Share of the width of a box inside the frame."""
    x0, _, x1, _ = box[:4]
    return max(0.0, min(x1, rect[2]) - max(x0, rect[0])) / max(x1 - x0, 1)


def _checks(name, rect, boxes, wanted, th, singer=None, singing=False):
    """Problems of one picture: list of (kind, message, value)."""
    fx0, fy0, fx1, fy1 = rect
    fw, fh = fx1 - fx0, fy1 - fy0
    out = []
    shown = {n: b for n, b in boxes.items() if _inside(b, rect) >= 0.6 and b[1] < fy1}
    tops = {n: b[1] for n, b in shown.items()}
    for n, t in tops.items():
        if t < fy0:
            out.append(("tête", f"tête de {n} coupée par le bord haut", (t - fy0) / fh))
        elif (t - fy0) / fh < th.head_top:
            out.append(("tête", f"tête de {n} trop près du bord haut", (t - fy0) / fh))
    if tops and (min(tops.values()) - fy0) / fh > th.air:
        out.append(
            ("tête", "trop d'air au-dessus des têtes", (min(tops.values()) - fy0) / fh)
        )
    group = [n for n in wanted if n in shown]
    base = base_name(name)
    if base == WIDE:
        group = list(shown)
    if group:
        # empty space: up to the first thing seen on that side (a neighbour in the frame
        # is not empty space)
        seen_ = [b for b in boxes.values() if b[2] > fx0 and b[0] < fx1 and b[1] < fy1]
        left = (min(max(b[0], fx0) for b in seen_) - fx0) / fw
        right = (fx1 - max(min(b[2], fx1) for b in seen_)) / fw
        if base == WIDE:
            limit = th.wide_side
        elif len(wanted) > 1:
            limit = th.group_side
        else:
            limit = None
        if limit is not None:
            if left > limit:
                out.append(("marge", "vide à gauche", left))
            if right > limit:
                out.append(("marge", "vide à droite", right))
        elif len(wanted) == 1 and wanted[0] in shown:
            b = shown[wanted[0]]
            off = ((b[0] + b[2]) / 2 - (fx0 + fx1) / 2) / fw
            if abs(off) > th.alone_offset:
                out.append(("marge", f"{wanted[0]} décentré", off))
    tight = base.endswith(" serré")  # (a tight shot need not show its person whole)
    for n, b in boxes.items():
        if tight and n in wanted:
            continue
        # on a shot of one person, a small piece of a neighbour is accepted (rule 3.1)
        lo, hi = th.cut_alone if len(wanted) == 1 and n not in wanted else th.cut
        for edge, left in ((fx0, True), (fx1, False)):
            if b[0] < edge < b[2]:
                inside = (b[2] - edge if left else edge - b[0]) / (b[2] - b[0])
                if lo < inside < hi:
                    out.append(("coupé", f"{n} coupé par un bord", inside))
                    break
        kp = b[4]
        if seen(kp, 5) and seen(kp, 6) and fx0 < shoulders_x(kp) < fx1:
            sy = (kp[5][1] + kp[6][1]) / 2
            hx, ey = head_x(kp), eyes_y(kp)
            out_top = ey is not None and ey < fy0
            out_side = hx is not None and not fx0 < hx < fx1
            if fy0 < sy < fy1 and (out_top or out_side):
                out.append(("coupé", f"corps de {n} sans sa tête", 0.0))
    if singer and singing:
        if singer not in shown:
            out.append(("chanteur", f"{singer} pas à l'image alors qu'il chante", 0.0))
        else:
            kp = shown[singer][4]
            hx, d = head_x(kp), face_direction(kp)
            if hx is not None and d is not None:
                s = head_scale(kp)
                if (hx - s - fx0) / fw < th.singer_edge and d < -th.gaze:
                    out.append(
                        ("chanteur", f"{singer} au bord gauche, tourné vers lui", d)
                    )
                if (fx1 - hx - s) / fw < th.singer_edge and d > th.gaze:
                    out.append(
                        ("chanteur", f"{singer} au bord droit, tourné vers lui", d)
                    )
    return out


def shot_alarms(
    shots: list[tuple[float, float, str]],
    framings: dict[str, tuple],
    frames: list[dict],
    order: list[str],
    singer: str | None = None,
    sings: Callable[[float, float], float] | None = None,
    guitarists: tuple[str, ...] = (),
    extend: dict[str, Extent] | None = None,
    fixed: dict[str, tuple[str, tuple[float, float, float, float]]] | None = None,
    accents: list[float] | np.ndarray | None = None,
    attacks: list[float] | np.ndarray | None = None,
    offset: float = 0.0,
    thresholds: Thresholds | None = None,
    variants: dict[str, list[tuple[float, float]]] | None = None,
) -> list[Alarm]:
    """Alarms of a montage, shot by shot.

    Parameters
    ----------
    shots : list[tuple[float, float, str]]
        ``(start, end, framing)`` in seconds of the montage (:func:`plan`, or
        :func:`mediautils.audit.montage_shots` on a montage project).
    framings : dict[str, tuple]
        The framings (:func:`mediautils.catalogue.catalogue`, with their variants); a
        shot whose name is not there is the whole picture.
    frames : list[dict]
        Named keypoints of the pictures of the rush (:func:`mediautils.identify.identify`).
    order : list[str]
        The musicians from left to right (the axis of the framings).
    singer : str, optional
        The singer (``"Fanfan"``).
    sings : callable, optional
        ``sings(a, b)``: share of the time between `a` and `b` where the singer sings
        (:meth:`mediautils.voice.Voice.sings`).
    guitarists : tuple[str, ...]
        People whose hands must be in the frame.
    extend : dict[str, Extent], optional
        What must be seen beyond the bodies (instruments), as for
        :func:`mediautils.catalogue.usual_places`.
    fixed : dict, optional
        Fixed things placed once per song: ``name -> (owner, (x0, y0, x1, y1))``, e.g.
        ``{"toms": ("Laetitia", (...))}``; flagged when cut by the frame of a shot of
        their owner.
    accents : array-like, optional
        Accents of the band (seconds): an accent at a cut in the middle of a phrase
        justifies it.
    attacks : array-like, optional
        Attacks (drum onsets, entries...; seconds): a cut far from all of them is
        flagged.
    offset : float
        Time of the montage = time of the pictures + `offset`.
    thresholds : Thresholds, optional
        The starting points.
    variants : dict, optional
        Variants of the catalogue for the parts of a song where the musicians stand
        elsewhere: ``name -> [(start, end)]`` (times of the pictures), for the framings
        named ``"<framing> (<name>)"``. Whether someone moved is judged against their
        usual place in these parts (else in the whole song).

    Returns
    -------
    list[Alarm]
        Sorted by time.

    Examples
    --------
    Fanfan sings all along; a shot on Thomas alone, then a shot on Thomas again,
    hardly tighter (and the cut falls in the middle of a phrase):

        >>> def body(x):
        ...     kp = [[x, 300, 0.9], [x - 10, 290, 0.9], [x + 10, 290, 0.9], [x + 30, 300, 0.9], [x - 30, 300, 0.9]]
        ...     kp += [[x - 80, 400, 0.9], [x + 80, 400, 0.9]] + [[0, 0, 0]] * 2 + [[x - 90, 600, 0.9]] * 2
        ...     return kp + [[0, 0, 0]] * 6
        >>> frames = [{"t": k / 2, "people": [{"kp": body(400), "name": "Thomas"}, {"kp": body(1300), "name": "Fanfan"}]}
        ...           for k in range(20)]
        >>> framings = {"Thomas": ("L", 400, 450, 2.0), "Thomas serré": ("L", 400, 420, 2.2)}
        >>> alarms = shot_alarms([(0, 5, "Thomas"), (5, 10, "Thomas serré")], framings, frames, ["Thomas", "Fanfan"],
        ...                      singer="Fanfan", sings=lambda a, b: 1.0)
        >>> for a in alarms:
        ...     print(clock(a.start), a.shot, ":", a.kind, ":", a.message)
        0:00,0 Thomas : chanteur : Fanfan pas à l'image alors qu'il chante (100% des images)
        0:05,0 Thomas serré : chanteur : Fanfan pas à l'image alors qu'il chante (100% des images)
        0:05,0 Thomas -> Thomas serré : coupe : coupe au milieu d'une phrase, sans respiration ni accent
        0:05,0 Thomas -> Thomas serré : enchaînement : deux cadrages trop proches (mêmes personnes, zoom x1.10)
    """
    th = thresholds or Thresholds()
    extend = extend or {}
    times = np.array([f["t"] + offset for f in frames])
    usuals = {None: _usual(frames, order)}
    for v, ivs in (variants or {}).items():
        part = [f for f in frames if any(a <= f["t"] < b for a, b in ivs)]
        usuals[v] = _usual(part, order)
    alarms = []
    for start, end, name in shots:
        variant = re.search(r"\(([^)]*)\)\s*$", name)
        usual = usuals.get(variant.group(1) if variant else None, usuals[None])
        framing = framings.get(name, ("?", W / 2, H / 2, 1.0))
        rect = framing_rect(framing)
        fw = rect[2] - rect[0]
        wanted = intended(name, order)
        ks = np.flatnonzero((times >= start) & (times < end))
        singing = bool(singer and sings and sings(start, end) > th.sings)
        found, pics = {}, {}
        for k in ks:
            boxes = _boxes(frames[k], extend)
            probs = _checks(name, rect, boxes, wanted, th, singer, singing)
            for n in wanted:
                if n not in boxes:
                    probs.append(("musicien", f"{n} caché ou pas trouvé", 0.0))
            for p in frames[k]["people"]:
                if p.get("name") is None and head_x(p["kp"]) is not None:
                    hx, ht = head_x(p["kp"]), head_top(p["kp"])
                    if (
                        rect[0] < hx < rect[2]
                        and ht is not None
                        and rect[1] < ht < rect[3]
                    ):
                        probs.append(
                            (
                                "musicien",
                                "quelqu'un de non identifié dans le cadre",
                                0.0,
                            )
                        )
                        break
            for n in guitarists:
                if n in wanted and n in boxes:
                    kp = boxes[n][4]
                    if any(
                        seen(kp, i)
                        and not (
                            rect[0] <= kp[i][0] <= rect[2]
                            and rect[1] <= kp[i][1] <= rect[3]
                        )
                        for i in WRISTS
                    ):
                        probs.append(("instrument", f"mains de {n} hors du cadre", 0.0))
            for kind, msg, v in probs:
                found.setdefault((kind, msg), []).append(v)
                pics.setdefault((kind, msg), []).append(float(times[k]))
        for (kind, msg), vs in found.items():
            if len(vs) >= max(1, th.share * len(ks)):
                share = len(vs) / len(ks)
                alarms.append(
                    Alarm(
                        start,
                        end,
                        name,
                        kind,
                        f"{msg} ({share:.0%} des images)",
                        pics[kind, msg],
                    )
                )
        for n in wanted:  # moved from their usual place
            xs = [shoulders_x(p["kp"]) for k in ks if (p := who(frames[k], n))]
            xs = [x for x in xs if x is not None]
            if xs and n in usual and abs(np.median(xs) - usual[n]) / fw > th.moved:
                alarms.append(
                    Alarm(
                        start,
                        end,
                        name,
                        "musicien",
                        f"{n} s'est déplacé ({np.median(xs) - usual[n]:+.0f} px)",
                    )
                )
        for thing, (owner, (x0, y0, x1, y1)) in (fixed or {}).items():
            if owner in wanted:
                iw = max(0.0, min(x1, rect[2]) - max(x0, rect[0]))
                ih = max(0.0, min(y1, rect[3]) - max(y0, rect[1]))
                share = iw * ih / max((x1 - x0) * (y1 - y0), 1)
                if 0.05 < share < 0.95:
                    alarms.append(
                        Alarm(
                            start,
                            end,
                            name,
                            "instrument",
                            f"{thing} de {owner} coupé ({share:.0%} dans le cadre)",
                        )
                    )
    alarms += _chain_alarms(shots, framings, order, th)
    alarms += _cut_alarms(shots, sings, accents, attacks, th)
    alarms.sort(key=lambda a: (a.start, a.kind))
    return alarms


def _usual(frames, order):
    """Usual shoulder position of each person (the densest place)."""
    from mediautils.catalogue import _mode

    out = {}
    for n in order:
        xs = [shoulders_x(p["kp"]) for f in frames if (p := who(f, n))]
        xs = [x for x in xs if x is not None]
        if xs:
            out[n] = _mode(xs, 60.0)
    return out


def _iou(a, b):
    ra, rb = framing_rect(a), framing_rect(b)
    iw = max(0.0, min(ra[2], rb[2]) - max(ra[0], rb[0]))
    ih = max(0.0, min(ra[3], rb[3]) - max(ra[1], rb[1]))
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(ra) + area(rb) - inter)


def _chain_alarms(shots, framings, order, th):
    out = []
    for (_, _, a), (t, end, b) in zip(shots, shots[1:]):
        if a not in framings or b not in framings or framings[a][0] != framings[b][0]:
            continue
        fa, fb = framings[a], framings[b]
        same = intended(a, order) == intended(b, order) and intended(a, order)
        ratio = max(fa[3], fb[3]) / min(fa[3], fb[3])
        if same and fb[3] < fa[3] and ratio >= th.zoom_ratio:
            out.append(
                Alarm(
                    t,
                    t,
                    f"{a} -> {b}",
                    "enchaînement",
                    "recul vers un cadrage plus large des mêmes personnes",
                )
            )
        if same and ratio < th.zoom_ratio:
            out.append(
                Alarm(
                    t,
                    t,
                    f"{a} -> {b}",
                    "enchaînement",
                    f"deux cadrages trop proches (mêmes personnes, zoom x{ratio:.2f})",
                )
            )
        elif WIDE not in (base_name(a), base_name(b)) and _iou(fa, fb) > th.iou:
            out.append(
                Alarm(
                    t,
                    t,
                    f"{a} -> {b}",
                    "enchaînement",
                    f"deux cadrages trop proches (recouvrement {_iou(fa, fb):.0%})",
                )
            )
    return out


def _cut_alarms(shots, sings, accents, attacks, th):
    out = []
    acc = np.asarray(accents if accents is not None else [], dtype=float)
    att = np.asarray(attacks if attacks is not None else [], dtype=float)
    for (_, _, a), (t, _, b) in zip(shots, shots[1:]):
        label = f"{a} -> {b}"
        if sings is not None and sings(t - th.phrase, t + th.phrase) >= 0.99:
            if not (len(acc) and np.min(np.abs(acc - t)) <= th.accent):
                out.append(
                    Alarm(
                        t,
                        t,
                        label,
                        "coupe",
                        "coupe au milieu d'une phrase, sans respiration ni accent",
                    )
                )
        if attacks is not None:
            d = float(np.min(np.abs(att - t))) if len(att) else float("inf")
            if d > th.attack:
                out.append(
                    Alarm(
                        t, t, label, "coupe", f"coupe loin de toute attaque ({d:.2f} s)"
                    )
                )
    return out


def catalogue_alarms(
    framings: dict[str, tuple],
    places: dict[str, Person],
    thresholds: Thresholds | None = None,
) -> list[tuple[str, str]]:
    """Alarms of a catalogue on the usual places of the musicians: ``(framing, message)``.

    The same checks as :func:`shot_alarms` for the heads, the margins and the people cut,
    on the boxes of :func:`mediautils.catalogue.usual_places` (instruments included).

        >>> from mediautils.catalogue import catalogue
        >>> places = {"A": Person(100, 700, 200, 260, 600, tight_hands=False),
        ...           "B": Person(1000, 1800, 250, 300, 650, tight_hands=False)}
        >>> z = catalogue(places, max_zoom=8)
        >>> catalogue_alarms(z, places)
        []
        >>> z["A"] = ("L", 900, 450, 1.6)  # moved by hand, too far right
        >>> catalogue_alarms(z, places)
        [('A', 'A décentré'), ('A', 'A coupé par un bord'), ('A', 'B coupé par un bord')]

    The very tight shots, held back by the largest zoom (quality of the picture):

        >>> catalogue_alarms(catalogue(places), places)[0]
        ('A très serré', 'pas franchement plus serré que « A serré » (zoom x1.19)')
    """
    th = thresholds or Thresholds()
    order = list(places)
    boxes = {}
    for n, p in places.items():
        cx = (p.x0 + p.x1) / 2
        kp = [[cx, p.eyes + 10, 0.0]] + [
            [0, 0, 0.0]
        ] * 16  # no keypoint seen: no head test
        boxes[n] = (p.x0, p.top, p.x1, p.bottom, kp)
    out = []
    for name, f in framings.items():
        for _, msg, _ in _checks(
            name, framing_rect(f), boxes, intended(name, order), th
        ):
            out.append((name, msg))
    for n in order:  # normal, tight, very tight: frankly different
        levels = [k for k in (n, f"{n} {TIGHT}", f"{n} {VERY_TIGHT}") if k in framings]
        for a, b in zip(levels, levels[1:]):
            ratio = framings[b][3] / framings[a][3]
            if ratio < th.zoom_ratio:
                out.append(
                    (b, f"pas franchement plus serré que « {a} » (zoom x{ratio:.2f})")
                )
    return out


def alarm_log(
    alarms: list[Alarm],
    path: str | os.PathLike,
    locate: Callable[[float], str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the alarms as a table where Claude notes, in one line each, what he did and
    why (``Décision``: corrigé / laissé; ``Pourquoi``).

    Parameters
    ----------
    alarms : list[Alarm]
        From :func:`shot_alarms`.
    path : str | os.PathLike
        The log (Markdown), in the song's work folder.
    locate : callable, optional
        ``locate(t) -> str``: where a time is in the song (e.g. ``"mesure 61, couplet 2"``,
        from the bar grid), added after the time.
    overwrite : bool
        Allow overwriting `path` (a log already filled in is not lost by mistake).

    Returns
    -------
    Path
        The log.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    lines = [
        "| Heure | Plan | Alarme | Décision | Pourquoi |",
        "|---|---|---|---|---|",
    ]
    for a in alarms:
        where = clock(a.start) + (f", {locate(a.start)}" if locate else "")
        lines.append(f"| {where} | {a.shot} | {a.kind} : {a.message} |  |  |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def draw_frame(
    image: str | os.PathLike,
    framing: tuple,
    dst: str | os.PathLike,
    frame: dict | None = None,
) -> Path:
    """The whole picture with the frame of a shot drawn on it (the margins are judged
    there), and the named skeletons if `frame` is given (:func:`mediautils.pose.detect_poses`
    picture, named). For Claude's checks only."""
    import cv2

    from mediautils.pose import SKELETON, read_image

    img = read_image(image)
    sx, sy = img.shape[1] / W, img.shape[0] / H
    x0, y0, x1, y1 = framing_rect(framing)
    cv2.rectangle(
        img,
        (int(x0 * sx), int(y0 * sy)),
        (int(x1 * sx), int(y1 * sy)),
        (0, 255, 255),
        4,
    )
    for p in (frame or {}).get("people", []):
        kp = p["kp"]
        col = (0, 255, 0) if p.get("name") else (180, 180, 180)
        for a, b in SKELETON:
            if seen(kp, a) and seen(kp, b):
                cv2.line(
                    img,
                    (int(kp[a][0] * sx), int(kp[a][1] * sy)),
                    (int(kp[b][0] * sx), int(kp[b][1] * sy)),
                    col,
                    3,
                )
        if seen(kp, 0):
            cv2.putText(
                img,
                p.get("name", "?"),
                (int(kp[0][0] * sx) - 30, int(kp[0][1] * sy) - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                col,
                2,
            )
    dst = Path(dst)
    cv2.imencode(dst.suffix or ".jpg", img)[1].tofile(str(dst))
    return dst
