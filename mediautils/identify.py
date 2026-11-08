"""Who is who among the people found by :func:`mediautils.pose.detect_poses`.

Light cues, combined, and the eye of Claude where they disagree (rules agreed with the
user, 2026-11-08; tested on *Sarbacane*, Burlingue, June 2026):

1. **the places on stage** (:class:`Zone`): people who never cross each other are told
   apart by where they stand, by their shoulders rather than their head, which leaves
   the zone when they lean;
2. **a distinctive sign**, chosen by eye once per song, for two people who cross each
   other (a cap against a bald skull: :func:`skull_lightness`; shorts against jeans...);
3. **who is in front** (:func:`in_front`), if the concert lends itself to it: the one in
   front looks bigger, has broader shoulders and their feet lower in the picture;
4. **continuity** from one picture to the next: a name follows its person; an isolated
   inversion of two people that cross each other is repaired from the neighbouring
   pictures (by their shoulders: in profile, the nose of the one in front passes over
   the head of the other).

When these cues contradict each other, the picture is flagged (:class:`Flag`) and
Claude looks at it. A face or silhouette recognition model would only come in if they
often failed; it has never been needed (and the user is told first).
"""

import copy
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mediautils.pose import (
    eyes_y,
    head_scale,
    head_x,
    read_image,
    same_body,
    seen,
    shoulders_x,
)


@dataclass
class Zone:
    """Where one or two people stand in the pictures of a camera, for a song.

    A detection is in the zone when the middle of its shoulders (else its nose) is
    between `x0` and `x1`, and its eyes between `y0` and `y1` (:func:`where`). A person
    may have several zones (a rectangle each): the first zone, in the order given, that
    finds them names them.

    Attributes
    ----------
    names : tuple[str, ...]
        One person (who never crosses anybody), or two who cross each other.
    x0, x1, y0, y1 : float
        Bounds of the zone (pixels of the pictures).
    sign : callable, optional
        For two people: ``sign(picture_lab, keypoints) -> value`` (e.g.
        :func:`skull_lightness`), lower for ``names[0]``, higher for ``names[1]``.
    threshold : float, optional
        Value of `sign` between the two people (decides alone when only one of them is
        found).
    front : str, optional
        Who of the two usually stands in front of the other, if the concert lends itself
        to it (Mathieu in front of Andrea, Burlingue June 2026). Without a sign, it
        decides; with a sign, a disagreement is flagged.
    """

    names: tuple[str, ...]
    x0: float = 0.0
    x1: float = math.inf
    y0: float = 0.0
    y1: float = math.inf
    sign: Callable | None = None
    threshold: float | None = None
    front: str | None = None

    def __post_init__(self):
        if len(self.names) not in (1, 2):
            raise ValueError("a zone holds one or two people")

    def holds(self, p: dict) -> bool:
        """Whether a detection is in the zone."""
        x, y = where(p)
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def where(p: dict) -> tuple[float, float]:
    """``(x, y)`` placing a detection in the zones: the middle of the shoulders (else the
    nose, else the middle of the box), and the eyes (else the ears, else the nose, even
    unsure: a back turned to the camera)."""
    kp, box = p["kp"], p["box"]
    x = shoulders_x(kp)
    if x is None:
        x = kp[0][0] if kp[0][2] > 0.05 else (box[0] + box[2]) / 2
    y = eyes_y(kp)
    if y is None:
        ears = [kp[i][1] for i in (3, 4) if seen(kp, i)]
        y = (
            float(np.mean(ears))
            if ears
            else float(kp[0][1] if kp[0][2] > 0.05 else box[1])
        )
    return float(x), float(y)


@dataclass
class Flag:
    """A picture where the cues disagree, for Claude to look at."""

    index: int
    t: float
    message: str


def score(p: dict) -> float:
    """Sum of the confidences of the keypoints of a detection."""
    return float(sum(q[2] for q in p["kp"]))


def who(frame: dict, name: str) -> dict | None:
    """The detection named `name` in a picture, or None."""
    return next((p for p in frame["people"] if p.get("name") == name), None)


def missing(frames: list[dict], name: str) -> list[int]:
    """Indices of the pictures where `name` is not found (e.g. for
    :func:`mediautils.pose.second_pass`)."""
    return [k for k, f in enumerate(frames) if who(f, name) is None]


def skull_lightness(lab: np.ndarray, kp: list) -> float | None:
    """Lightness (L of CIELAB, 0-255 in OpenCV) of the top of the skull: a bald skull
    under the stage lights is light, a cap or dark hair is dark.

    The sample is centred between the nose and the visible ears (in profile, the nose is
    at the edge of the head), above the eyes. On *Sarbacane*, 134 separated Andrea (cap)
    from Mathieu (bald). None if the eyes and the nose are not seen.

        >>> lab = np.zeros((600, 800, 3), np.uint8)
        >>> lab[150:280, 300:500, 0] = 200
        >>> kp = [[400, 300, 0.9], [390, 290, 0.9], [410, 290, 0.9]] + [[0, 0, 0]] * 14
        >>> skull_lightness(lab, kp)
        200.0
    """
    x, ey = head_x(kp), eyes_y(kp)
    if x is None or ey is None:
        return None
    s = head_scale(kp)
    h, w = lab.shape[:2]
    y0, y1 = int(max(ey - 1.2 * s, 0)), int(max(ey - 0.5 * s, 0))
    x0, x1 = int(max(x - 0.5 * s, 0)), int(min(x + 0.5 * s, w))
    crop = lab[y0:y1, x0:x1, 0]
    return float(np.median(crop)) if crop.size else None


def in_front(a: list, b: list) -> int:
    """Which of two bodies is in front: 1 for `a`, -1 for `b`, 0 if unclear.

    The one in front looks bigger: a longer torso (shoulders to hips, by 10 % at
    least; unlike the width of the shoulders, it does not shrink when someone turns
    sideways or shows their back: Mathieu, *Sarbacane*); and, when the feet of both are
    seen, lower feet. The cues must not disagree.

        >>> def body(torso, feet):
        ...     kp = [[0, 0, 0]] * 5 + [[500, 300, 1], [700, 300, 1]] + [[0, 0, 0]] * 4
        ...     return kp + [[500, 300 + torso, 1], [700, 300 + torso, 1]] + [[0, 0, 0]] * 2 + [[0, feet, 1]] * 2
        >>> in_front(body(300, 1000), body(250, 950)), in_front(body(250, 950), body(300, 1000))
        (1, -1)
        >>> in_front(body(300, 900), body(250, 1000))
        0
    """

    def torso(kp):
        if not all(seen(kp, i) for i in (5, 6, 11, 12)):
            return None
        return (kp[11][1] + kp[12][1] - kp[5][1] - kp[6][1]) / 2

    def feet(kp):
        ys = [kp[i][1] for i in (15, 16) if seen(kp, i)]
        return max(ys) if ys else None

    votes = []
    wa, wb = torso(a), torso(b)
    if wa and wb and wa > 0 and wb > 0 and max(wa, wb) > 1.1 * min(wa, wb):
        votes.append(1 if wa > wb else -1)
    fa, fb = feet(a), feet(b)
    if fa is not None and fb is not None and abs(fa - fb) > 20:
        votes.append(1 if fa > fb else -1)
    if not votes or len(set(votes)) > 1:
        return 0
    return votes[0]


def _dedupe(people):
    """Two skeletons on the same body: the best one is kept (the order is kept)."""
    kept = []
    for p in sorted(people, key=score, reverse=True):
        if not any(same_body(p["kp"], q["kp"]) for q in kept):
            kept.append(p)
    return [p for p in people if any(p is q for q in kept)]


def identify(
    frames: list[dict],
    zones: list[Zone],
    images: list[str | os.PathLike] | None = None,
    max_eyes: float | None = None,
) -> tuple[list[dict], list[Flag]]:
    """Name the people of every picture.

    Parameters
    ----------
    frames : list[dict]
        Keypoints of the pictures (:func:`mediautils.pose.detect_poses`, possibly
        completed by :func:`mediautils.pose.second_pass`); not modified.
    zones : list[Zone]
        The zones of the song, chosen by eye on a few pictures.
    images : list, optional
        The pictures; needed when a zone has a `sign`.
    max_eyes : float, optional
        Detections whose eyes are lower than this are not named (a piece of a body
        without its head).

    Returns
    -------
    tuple[list[dict], list[Flag]]
        A copy of `frames` where the detections found carry their ``"name"`` (and
        ``"sign"``, the value of the sign of their zone); and the pictures to look at.

    Examples
    --------
    Thomas on the left, alone in his zone; Andrea and Mathieu cross each other on the
    right, Mathieu in front:

        >>> def body(x, w, feet=None):
        ...     kp = [[x, 300, 0.9], [x - 10, 290, 0.9], [x + 10, 290, 0.9]] + [[0, 0, 0]] * 2
        ...     kp += [[x - w / 2, 400, 0.9], [x + w / 2, 400, 0.9]] + [[0, 0, 0]] * 8
        ...     return {"box": [x - w, 200, x + w, 1000], "kp": kp + [[x, feet or 0, 0.9 if feet else 0]] * 2}
        >>> zones = [Zone(("Thomas",), 0, 500), Zone(("Andrea", "Mathieu"), 1250, front="Mathieu")]
        >>> frames = [{"t": 0.0, "people": [body(300, 180), body(1400, 150, 950), body(1600, 200, 1050)]},
        ...           {"t": 0.5, "people": [body(310, 180), body(1450, 210, 1060), body(1620, 160, 960)]}]
        >>> named, flags = identify(frames, zones)
        >>> [[p.get("name") for p in f["people"]] for f in named]
        [['Thomas', 'Andrea', 'Mathieu'], ['Thomas', 'Mathieu', 'Andrea']]

    The two cross between the two pictures (Mathieu passes to the left): nothing is
    flagged, both cues agree. An isolated inversion in a row of pictures would be
    repaired and flagged.
    """
    out = copy.deepcopy(frames)
    labs = {}

    def lab(k):
        if k not in labs:
            import cv2

            if images is None:
                raise ValueError("a zone with a sign needs the pictures")
            labs.clear()  # one picture at a time in memory
            labs[k] = cv2.cvtColor(read_image(images[k]), cv2.COLOR_BGR2LAB)
        return labs[k]

    flags = []
    for k, f in enumerate(out):
        f["people"] = _dedupe(f["people"])
        for p in f["people"]:
            p.pop("name", None)
            p.pop("sign", None)
        heads = [p for p in f["people"] if max_eyes is None or where(p)[1] <= max_eyes]
        for z in zones:
            if any(who(f, n) for n in z.names):  # found already by an earlier zone
                continue
            inside = sorted(
                (p for p in heads if "name" not in p and z.holds(p)),
                key=score,
                reverse=True,
            )[: len(z.names)]
            if not inside:
                continue
            if len(z.names) == 1:
                inside[0]["name"] = z.names[0]
                continue
            msg = _pair(z, inside, (lambda: lab(k)) if z.sign else None)
            if msg:
                flags.append(Flag(k, f["t"], msg))
    flags += _repair_inversions(out, [z for z in zones if len(z.names) == 2])
    flags.sort(key=lambda g: g.index)
    return out, flags


def _pair(z, inside, lab):
    """Name the (one or two) detections of a zone of two people; a message if the cues
    disagree."""
    a_name, b_name = z.names
    values = {}
    if lab is not None:
        img = lab()
        for p in inside:
            v = z.sign(img, p["kp"])
            if v is not None:
                values[id(p)] = v
                p["sign"] = round(float(v), 1)
    if len(inside) == 2:
        p, q = inside
        front = in_front(p["kp"], q["kp"])
        if id(p) in values and id(q) in values:
            lo, hi = sorted(inside, key=lambda r: values[id(r)])
            lo["name"], hi["name"] = a_name, b_name
            msgs = []
            if z.threshold is not None and (
                (values[id(lo)] >= z.threshold) == (values[id(hi)] >= z.threshold)
            ):
                msgs.append("signe ambigu (les deux du même côté du seuil)")
            if z.front and front:
                ahead = p if front == 1 else q
                if ahead.get("name") != z.front:
                    msgs.append(f"le signe met {z.front} derrière")
            return " ; ".join(msgs)
        known = [r for r in inside if id(r) in values]
        if known and z.threshold is not None:  # the sign of one of them tells both
            r = known[0]
            other = q if r is p else p
            r["name"] = a_name if values[id(r)] < z.threshold else b_name
            other["name"] = b_name if r["name"] == a_name else a_name
            if z.front and front:
                ahead = p if front == 1 else q
                if ahead.get("name") != z.front:
                    return f"le signe met {z.front} derrière"
            return ""
        if z.front and front:
            ahead, behind = (p, q) if front == 1 else (q, p)
            ahead["name"] = z.front
            behind["name"] = b_name if z.front == a_name else a_name
            return ""
        # nothing tells them apart here: by their order, left to right, decided by continuity
        p, q = sorted(inside, key=lambda r: shoulders_x(r["kp"]))
        p["name"], q["name"] = a_name, b_name
        return "aucun indice ne les sépare (ordre gauche-droite supposé)"
    (p,) = inside
    if id(p) in values and z.threshold is not None:
        p["name"] = a_name if values[id(p)] < z.threshold else b_name
        return ""
    return f"une seule personne dans la zone {a_name}-{b_name}, sans signe : non nommée"


def _repair_inversions(frames, zones):
    """Two people of a zone in the inverse order of their neighbours (2 pictures before and
    after, by the shoulders), on one picture only: swapped back."""
    flags = []
    for z in zones:
        a_name, b_name = z.names
        pairs = [k for k, f in enumerate(frames) if who(f, a_name) and who(f, b_name)]

        def order(k):
            a, b = who(frames[k], a_name), who(frames[k], b_name)
            xa, xb = shoulders_x(a["kp"]), shoulders_x(b["kp"])
            return 0 if xa is None or xb is None else (1 if xa < xb else -1)

        orders = {k: order(k) for k in pairs}
        for j, k in enumerate(pairs):
            near = pairs[max(j - 2, 0) : j] + pairs[j + 1 : j + 3]
            if (
                orders[k]
                and len(near) >= 2
                and all(orders[n] == -orders[k] for n in near)
                and all(abs(n - k) <= 4 for n in near)
            ):
                a, b = who(frames[k], a_name), who(frames[k], b_name)
                a["name"], b["name"] = b_name, a_name
                flags.append(
                    Flag(
                        k, frames[k]["t"], f"inversion isolée {a_name}/{b_name} réparée"
                    )
                )
    return flags


def swap_names(frames: list[dict], index: int, a: str, b: str) -> None:
    """Swap two names on one picture (a correction by eye), in place."""
    pa, pb = who(frames[index], a), who(frames[index], b)
    if pa is not None:
        pa["name"] = b
    if pb is not None:
        pb["name"] = a


def gaze_series(frames: list[dict], name: str) -> list[tuple[float, float | None]]:
    """``(t, face direction)`` of a person on every picture where they are found
    (:func:`mediautils.pose.face_direction`)."""
    from mediautils.pose import face_direction

    return [(f["t"], face_direction(p["kp"])) for f in frames if (p := who(f, name))]
