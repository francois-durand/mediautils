"""Catalogue of the framings of a camera for a song, from the usual places of the musicians.

Rules agreed with the user (2026-11-08, ``Règles de montage.md``, sections 3 and 8):

- **content**: a framing for every interval of the axis of the musicians (their order
  in the picture, left to right), the person alone included, plus a **tight** and a
  **very tight** shot of every person; for five musicians, 25 framings per camera. The
  whole catalogue is computed (it costs nothing): an intention asked by the user during
  the back-and-forth is already there;
- **why a catalogue** rather than frames shot by shot: the same intention keeps the same
  frame from one round to the next; it is light; a shot changes only by decision;
- **normal, tight and very tight are heights of the body**, what each one shows, not
  zoom ratios set so that a chain of two of them shows (the user, 2026-11-09: that rule
  had made the very tight shot of the singer far too tight). For the singer (Fanfan):
  normal a little under the belt, tight a little above it, very tight a little under the
  elbows (*Cocaine*, June 2026: zooms 1.9, 2.4, 3.0); placed with ``fixed``. For a
  guitarist it is very different (hands and neck); the ratios ``tight`` and
  ``very_tight`` are only the default for the others;
- **the places come from where each one usually stands in the song**
  (:func:`usual_places`): not the mean, which a short move distorts, nor the envelope of
  their moves, which gives frames too wide;
- **two almost identical framings are both kept** (the contrast rule prevents chaining
  them); they are often the sign of an ambiguous intention, which Claude corrects;
- **what to show, then a margin** (the method the user also follows to crop his photos):
  no line of thirds, no golden ratio (the eyes on the third line leave too much air above
  the heads); the ceiling goes first (it is seldom of interest); the wide shot shows
  from the first musician to the last with a small margin (3 %), the feet kept.

The framings computed here are proposals: Claude looks at them, places by eye what
must be (``fixed``), and the alarms (:mod:`mediautils.alarms`) check them; nothing moves
a framing by itself. The instruments are not keypoints: what must be seen of a guitar
(neck, bridge), of the drums, is added by eye to each place (``extend``).

A framing is ``(camera, centre_x, centre_y, zoom)`` in pixels of the 1920 x 1080 source,
as in :func:`mediautils.montage.build_montage`.
"""

import re
from dataclasses import dataclass

import numpy as np

from mediautils.framing import Person
from mediautils.identify import who
from mediautils.pose import (
    HIPS,
    WRISTS,
    body_box,
    eyes_y,
    head_scale,
    head_top,
    seen,
    shoulders_x,
)

W, H = 1920, 1080
TIGHT = "serré"
VERY_TIGHT = "très serré"
WIDE = "large"


@dataclass
class Extent:
    """What must be seen beyond the body of a musician, placed by eye (pixels): the neck of
    a guitar, its bridge, the drums.

    Attributes
    ----------
    left, right, below : float
        Room added to the left, to the right and below the body (hands).
    alone : float, optional
        Room added below in the shots of the person alone only (the drums: the shots of
        the drummer show them, the groups need not).
    tight_hands : bool
        Whether the tight shots keep the hands (an instrumentalist) or not (the singer:
        the head and the microphone).
    """

    left: float = 0.0
    right: float = 0.0
    below: float = 0.0
    alone: float | None = None
    tight_hands: bool = True


@dataclass
class Place(Person):
    """Usual place of a musician (:func:`usual_places`): a :class:`~mediautils.framing.Person`
    whose `bottom` includes the instrument (:class:`Extent`), and `hands` the bottom of
    the lowest hand, as low as it usually goes (what the tight shots of an
    instrumentalist keep: :func:`hand_bottom`)."""

    hands: float | None = None


def hand_bottom(kp: list) -> float | None:
    """Bottom of the lowest hand (pixels): the keypoint is the wrist, the hand goes on
    below it (taken as 1.5 half-widths of the head). None if no wrist is seen.

    *People have the power* (Burlingue, 2026-11-08): tight shots that kept the wrists
    only cut Mathieu's strumming hand.

        >>> kp = [[0, 0, 0]] * 5 + [[400, 400, 0.9], [600, 400, 0.9]] + [[0, 0, 0]] * 2
        >>> kp += [[390, 600, 0.9], [610, 640, 0.9]] + [[0, 0, 0]] * 6
        >>> hand_bottom(kp)
        715.0
    """
    ys = [kp[i][1] for i in WRISTS if seen(kp, i)]
    return max(ys) + 1.5 * head_scale(kp) if ys else None


def _mode(xs, spread):
    """The value around which most of `xs` lie (densest window of width 2 * spread)."""
    xs = np.sort(np.asarray(xs, dtype=float))
    counts = np.searchsorted(xs, xs + spread, side="right") - np.searchsorted(
        xs, xs - spread, side="left"
    )
    best = xs[counts == counts.max()]
    return float(np.median(best))


def usual_places(
    frames: list[dict],
    names: list[str],
    extend: dict[str, Extent] | None = None,
    spread: float = 60.0,
    since: float | None = None,
    until: float | None = None,
) -> dict[str, "Place"]:
    """Where each musician usually stands, as the box of :class:`~mediautils.framing.Person`
    (a :class:`Place`).

    For each person, the pictures where their shoulders are within `spread` pixels of
    the place where they stand most of the time; on these, the median of their body
    (keypoints: from the top of the skull to the hands, between the extreme points),
    widened by their :class:`Extent`.

    Parameters
    ----------
    frames : list[dict]
        Named keypoints (:func:`mediautils.identify.identify`).
    names : list[str]
        The musicians.
    extend : dict[str, Extent], optional
        What must be seen beyond the body of each one (instruments), placed by eye.
    spread : float
        Half-width of the usual place (pixels).
    since, until : float, optional
        Only the pictures between these times (a variant of the catalogue for the part
        of a song where someone stands elsewhere).

    Returns
    -------
    dict[str, Person]
        The places, **in the order of the axis** (left to right); a person never found
        is left out.

    Examples
    --------
        >>> def body(x):
        ...     kp = [[x, 300, 0.9], [x - 10, 290, 0.9], [x + 10, 290, 0.9]] + [[0, 0, 0]] * 2
        ...     kp += [[x - 80, 400, 0.9], [x + 80, 400, 0.9]] + [[0, 0, 0]] * 2
        ...     return kp + [[x - 90, 600, 0.9], [x + 90, 620, 0.9]] + [[0, 0, 0]] * 6
        >>> frames = [{"t": k / 2, "people": [{"kp": body(1000 if k < 9 else 1500), "name": "B"},
        ...                                    {"kp": body(400), "name": "A"}]} for k in range(10)]
        >>> places = usual_places(frames, ["B", "A"], extend={"A": Extent(left=100)})
        >>> list(places)
        ['A', 'B']
        >>> places["B"].x0, places["B"].x1, places["B"].top, places["B"].bottom
        (910.0, 1090.0, 230.0, 620.0)
        >>> places["A"].x0
        210.0
    """
    extend = extend or {}
    out = {}
    for name in names:
        rows = []
        for f in frames:
            if (
                since is not None
                and f["t"] < since
                or until is not None
                and f["t"] >= until
            ):
                continue
            p = who(f, name)
            if p is None:
                continue
            kp = p["kp"]
            x, box, top, e = shoulders_x(kp), body_box(kp), head_top(kp), eyes_y(kp)
            if x is None or box is None or top is None:
                continue
            hand = hand_bottom(kp)
            low = (
                [kp[i][1] for i in WRISTS if seen(kp, i)]
                or [kp[i][1] for i in HIPS if seen(kp, i)]
                or [box[3]]
            )
            rows.append((x, box[0], box[2], top, e, max(low), hand or np.nan))
        if not rows:
            continue
        rows = np.array(rows, dtype=float)
        mode = _mode(rows[:, 0], spread)
        rows = rows[np.abs(rows[:, 0] - mode) <= spread]
        x, x0, x1, top, e, low = np.median(rows[:, :6], axis=0)
        # the hands: as low as they usually go (a strumming hand moves), not the median
        hands = rows[:, 6][~np.isnan(rows[:, 6])]
        hand = float(np.percentile(hands, 90)) if len(hands) else float(low)
        ext = extend.get(name, Extent())
        out[name] = (
            x,
            Place(
                x0=float(x0 - ext.left),
                x1=float(x1 + ext.right),
                top=float(top),
                eyes=float(e),
                bottom=float(low + ext.below),
                alone=None if ext.alone is None else float(low + ext.alone),
                tight_hands=ext.tight_hands,
                hands=hand,
            ),
        )
    return {n: p for n, (_, p) in sorted(out.items(), key=lambda item: item[1][0])}


def framing_rect(framing: tuple) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` of a framing in the source picture, kept inside it as
    :func:`~mediautils.montage.build_montage` does.

        >>> framing_rect(("L", 960, 540, 2.0))
        (480.0, 270.0, 1440.0, 810.0)
        >>> framing_rect(("L", 100, 540, 2.0))
        (0.0, 270.0, 960.0, 810.0)
    """
    _, cx, cy, z = framing[:4]
    w, h = W / z, H / z
    x0 = min(max(cx - w / 2, 0), W - w)
    y0 = min(max(cy - h / 2, 0), H - h)
    return float(x0), float(y0), float(x0 + w), float(y0 + h)


def _place(x0, x1, top, bottom, margin, headroom, max_zoom, min_h=0.0):
    """Frame (cx, cy, zoom): what to show, then a margin; the ceiling goes first."""
    w = (x1 - x0) / (1 - 2 * margin)
    h = max(w * H / W, (bottom - top) / (1 - headroom - margin), H / max_zoom, min_h)
    h = min(h, H)
    w = h * W / H
    cx = min(max((x0 + x1) / 2, w / 2), W - w / 2)
    y0 = min(max(top - headroom * h, 0), H - h)
    return float(cx), float(y0 + h / 2), float(H / h)


def catalogue(
    places: dict[str, Person],
    camera: str = "L",
    margin: float = 0.04,
    headroom: float = 0.10,
    wide_margin: float = 0.03,
    tight: float = 1.45,
    very_tight: float = 1.45,
    max_zoom: float = 4.0,
    fixed: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, tuple[str, float, float, float]]:
    """Every framing of a camera: the intervals of the axis, and a tight and a very tight
    shot of every person.

    Parameters
    ----------
    places : dict[str, Person]
        The musicians seen by the camera, **from left to right** (:func:`usual_places`).
    camera : str
        Name of the camera in the framings.
    margin : float
        Margin around what is shown, on the sides and below, as a share of the frame.
    headroom : float
        Air above the highest head, as a share of the frame height (neither "head
        cutter" nor too much air, which cuts what matters below).
    wide_margin : float
        Margin of the wide shot on the sides (3 % was right at Burlingue, June 2026).
    tight, very_tight : float
        Zoom of the tight shot relative to the person alone, of the very tight one
        relative to the tight one (frankly different: a chain of two framings of the same
        people needs a zoom ratio of 1.4 at least).
    max_zoom : float
        Largest zoom, tight shots included (quality of the picture; a 4K rush allows
        more than a 1080p one). A tight shot held back by it may not be frankly tighter:
        :func:`mediautils.alarms.catalogue_alarms` flags it.
    fixed : dict, optional
        Framings placed by eye, ``name -> (centre_x, centre_y, zoom)``: they replace the
        computed ones of the same name, the others (variants) are added at the end.

    Returns
    -------
    dict[str, tuple[str, float, float, float]]
        ``name -> (camera, centre_x, centre_y, zoom)``, in the order of the axis:
        ``"large"``, then for each person the intervals that start with them from the
        largest, the person alone, ``"<name> serré"``, ``"<name> très serré"``. Groups
        are named ``"Thomas-Laetitia"`` (from Thomas to Laetitia, everybody between them
        included).

    Examples
    --------
    Three singers (their tight shots need not keep their hands):

        >>> def singer(x0, x1, top, eyes, bottom):
        ...     return Person(x0, x1, top, eyes, bottom, tight_hands=False)
        >>> places = {"A": singer(100, 500, 200, 260, 600), "B": singer(800, 1100, 250, 300, 650),
        ...           "C": singer(1300, 1800, 200, 260, 700)}
        >>> z = catalogue(places)
        >>> len(z), list(z)[:6]
        (12, ['large', 'A-B', 'A', 'A serré', 'A très serré', 'B-C'])
        >>> cam, cx, cy, zoom = z["large"]
        >>> round(cx), round(zoom, 2), round(cy + 540 / zoom)  # the feet kept: down to the bottom
        (950, 1.06, 1080)
        >>> round(z["B serré"][3] / z["B"][3], 2), round(z["B très serré"][3] / z["B serré"][3], 2)
        (1.45, 1.19)

    The very tight shot of B is held back by the largest zoom (4 by default):

        >>> round(z["B très serré"][3], 2)
        4.0
        >>> framing_rect(z["B"])[1] < 250  # the head inside, with some air
        True

    An instrumentalist keeps both hands in his tight shots: when his place is only his
    body down to the hands, they cannot be tighter (an alarm says so); with the guitar
    below the hands (:func:`usual_places`, :class:`Extent`), they drop the guitar.

        >>> z = catalogue({"G": Place(800, 1100, 250, 300, 800, hands=650)})
        >>> round(z["G"][3], 2), round(z["G serré"][3], 2)
        (1.69, 2.3)
        >>> framing_rect(z["G serré"])[3] >= 650
        True
    """
    names = list(places)
    n = len(names)
    ps = [places[k] for k in names]
    out = {}
    for i in range(n):
        for j in range(n - 1, i - 1, -1):
            ms = ps[i : j + 1]
            x0, x1 = min(p.x0 for p in ms), max(p.x1 for p in ms)
            top, bottom = min(p.top for p in ms), max(p.bottom for p in ms)
            if i == 0 and j == n - 1 and n > 1:
                w = (x1 - x0) / (1 - 2 * wide_margin)
                h = min(max(w * H / W, (H - top) / (1 - headroom)), H)
                w = h * W / H
                cx = min(max((x0 + x1) / 2, w / 2), W - w / 2)
                out[WIDE] = (camera, float(cx), float(H - h / 2), float(H / h))
                continue
            if i == j and ms[0].alone is not None:
                bottom = max(bottom, ms[0].alone)
            cx, cy, z = _place(x0, x1, top, bottom, margin, headroom, max_zoom)
            name = names[i] if i == j else f"{names[i]}-{names[j]}"
            out[name] = (camera, cx, cy, z)
        p = ps[i]
        z = out[names[i]][3]
        # the tight shots of an instrumentalist keep both hands (quasi absolute, rule 3.4):
        # they zoom in less if needed (and may then not be frankly tighter: an alarm says so)
        hands = getattr(p, "hands", None)
        low = (p.bottom if hands is None else hands) if p.tight_hands else None
        for word, factor in ((TIGHT, tight), (VERY_TIGHT, tight * very_tight)):
            zt = min(z * factor, max_zoom)
            while True:
                h = H / zt
                y0 = min(max(p.top - headroom * h, 0), H - h)
                if low is None or y0 + h >= low + margin * h or zt <= z:
                    break
                zt = max(zt * 0.97, z)
            cx = min(max((p.x0 + p.x1) / 2, W / zt / 2), W - W / zt / 2)
            out[f"{names[i]} {word}"] = (
                camera,
                float(cx),
                float(y0 + h / 2),
                float(zt),
            )
    order = [WIDE] + [k for k in out if k != WIDE] if WIDE in out else list(out)
    out = {k: out[k] for k in order}
    for name, f in (fixed or {}).items():
        out[name] = (camera, *map(float, f))
    return out


_SUFFIX = re.compile(r"\s+(@.*|\(.*\))$")


def base_name(name: str) -> str:
    """Name of a framing without its variant (``"Fanfan @1:23,5"``, ``"Fanfan (fin)"``).

    >>> base_name("Fanfan-Laetitia @1:23,5"), base_name("Fanfan serré (2e partie)")
    ('Fanfan-Laetitia', 'Fanfan serré')
    """
    return _SUFFIX.sub("", name).strip()


def variant_name(name: str, t: float) -> str:
    """Name of the variant of a framing for the shot that starts at `t` (seconds).

    >>> variant_name("Fanfan", 83.5)
    'Fanfan @1:23,5'
    """
    m, s = divmod(t, 60)
    return f"{base_name(name)} @{int(m)}:{s:04.1f}".replace(".", ",")


def intended(name: str, order: list[str]) -> list[str]:
    """The people a framing is meant to show, from its name.

    >>> order = ["Thomas", "Fanfan", "Laetitia", "Andrea", "Mathieu"]
    >>> intended("Fanfan-Andrea @2:01,0", order), intended("Laetitia très serré", order)
    (['Fanfan', 'Laetitia', 'Andrea'], ['Laetitia'])
    >>> intended("large", order) == order, intended("public", order)
    (True, [])
    """
    base = base_name(name)
    for word in (VERY_TIGHT, TIGHT):
        if base.endswith(" " + word):
            base = base[: -len(word) - 1]
            break
    if base == WIDE:
        return list(order)
    ends = base.split("-")
    if not all(e in order for e in ends):
        return []
    a, b = order.index(ends[0]), order.index(ends[-1])
    return list(order[min(a, b) : max(a, b) + 1])
