"""Framings of a camera from the positions of the musicians.

The framings of a camera are the **intervals of the axis of the musicians**, as seen
by that camera (from left to right): each interval shows its people and everybody
between them; the whole interval is the wide shot. A person alone also has a
**tight** framing.

Each person is a box, placed by eye on a picture of the song: it covers the head, the
body and what must be seen of the instrument (hands, fretboard up to the bridge,
drumsticks). The framings follow by geometry:

- the eyes of the main person of the framing are on the line of the upper third (the
  tighter the shot, the closer the head comes to the top edge, as it should); nobody's
  head nor hands are cut;
- a person alone is centred (a shot on someone may cut the neighbours);
- for a group, an edge never falls in the middle of somebody: it falls in an empty
  space, or clearly includes the neighbour, or clearly excludes them (a small piece at
  the edge is tolerated); among the positions that do so, the one best centred on the
  interval (not too strict: if none, the least cut).

Two consecutive shots are "almost the same" when they show the same interval at the
same zoom level: that is decided when choosing the shots, from the names.
"""

import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

W, H = 1920, 1080


@dataclass
class Person:
    """Box of a musician in the picture of a camera (pixels of the 1920 x 1080 source).

    Attributes
    ----------
    x0, x1 : float
        Left and right of the person and of what must be seen of their instrument.
    top : float
        Top of the head.
    eyes : float
        Height of the eyes.
    bottom : float
        Lowest thing that must be seen (hands, drumsticks).
    """

    x0: float
    x1: float
    top: float
    eyes: float
    bottom: float


def _frame(people, members, main, margin, eye_line, tolerance, max_zoom=1e9):
    """Frame (cx, cy, zoom) showing `members` (indices in `people`)."""
    ms = [people[i] for i in members]
    x0, x1 = min(p.x0 for p in ms), max(p.x1 for p in ms)
    top, bottom = min(p.top for p in ms), max(p.bottom for p in ms)
    eyes = people[main].eyes
    w = (x1 - x0) * (1 + 2 * margin)
    # the eyes on the third line, or higher (up to a quarter) if that keeps the shot tighter;
    # the heads and the hands inside, with a margin
    best = None
    for e in np.linspace(eye_line, min(eye_line, 0.25), 9):
        he = max(
            w * H / W, (eyes - top) / (e - margin), (bottom - eyes) / (1 - e - margin)
        )
        if best is None or he < best[0] - 1:
            best = (he, e)
    h, e = best
    h = min(max(h, H / max_zoom), H)
    w = h * W / H
    y0 = min(max(eyes - e * h, 0), H - h)
    if y0 > top - margin * h:  # a head would be cut: raise the frame
        y0 = max(top - margin * h, 0)
    # horizontal position: the edges do not cut the others in the middle
    others = [p for k, p in enumerate(people) if k not in members]
    lo, hi = max(x1 + margin * w * 0.5 - w, 0), min(x0 - margin * w * 0.5, W - w)
    if lo > hi:  # the interval does not fit with its margins: centre it
        lo = hi = min(max((x0 + x1) / 2 - w / 2, 0), W - w)
    centre = min(max((x0 + x1) / 2 - w / 2, lo), hi)

    def cut(left):
        c = 0.0
        for p in others:
            for e in (left, left + w):
                if p.x0 < e < p.x1:
                    c = max(c, min(e - p.x0, p.x1 - e) / (p.x1 - p.x0))
        return c

    if (
        len(members) == 1
    ):  # a shot on one person may cut the neighbours: centred on them
        return centre + w / 2, y0 + h / 2, W / w
    # a group: not too far from the centre (a third of the frame at most)
    candidates = np.linspace(lo, hi, 200) if hi > lo else np.array([lo])
    candidates = candidates[abs(candidates - centre) <= w / 3]
    if not len(candidates):
        candidates = np.array([centre])
    ok = [c for c in candidates if cut(c) <= tolerance]
    left = min(ok, key=lambda c: abs(c - centre)) if ok else min(candidates, key=cut)
    return left + w / 2, y0 + h / 2, W / w


def interval_framings(
    people: dict[str, Person],
    camera: str = "L",
    main: str | None = None,
    margin: float = 0.04,
    eye_line: float = 1 / 3,
    tight: float = 1.35,
    tolerance: float = 0.15,
    no_tight: tuple[str, ...] = (),
    max_zoom: float = 2.4,
) -> dict[str, tuple[str, float, float, float]]:
    """Every framing of a camera: the intervals of its axis, and a tight shot per person.

    Parameters
    ----------
    people : dict[str, Person]
        The musicians seen by the camera, **from left to right** (people in the same
        line of sight count as one).
    camera : str
        Name of the camera in the zooms.
    main : str, optional
        Person whose eyes are on the third line when they are in the framing (the
        singer); otherwise the person whose head is the highest.
    margin : float
        Margin around what must be seen, as a share of the frame.
    eye_line : float
        Height of the eyes of the main person, as a share of the frame from its top.
    tight : float
        Zoom of a tight shot relative to the shot of the person alone.
    tolerance : float
        Share of a person's box an edge may cut (a piece at the edge is tolerated).
    no_tight : tuple[str, ...]
        People without a tight shot.
    max_zoom : float
        Largest zoom of a framing (picture quality: far, small people are not blown up);
        a tight shot may go `tight` times further.

    Returns
    -------
    dict[str, tuple[str, float, float, float]]
        ``name -> (camera, centre_x, centre_y, zoom)`` (the zooms of
        :func:`mediautils.build_montage`): ``"Fanfan"``, ``"Fanfan serré"``,
        ``"Thomas-Laetitia"`` (from Thomas to Laetitia), and ``"large"`` (everybody).
        A framing is named after the people it really shows: near the edge of the
        picture, a person alone may not be framed without a neighbour (no ``"Thomas"``
        then, only ``"Thomas-Fanfan"``); intervals that show the same people at about
        the same zoom give one framing.

    Examples
    --------
        >>> people = {"A": Person(100, 500, 200, 260, 600), "B": Person(800, 1100, 250, 300, 650),
        ...           "C": Person(1300, 1800, 200, 260, 700)}
        >>> z = interval_framings(people, main="B")
        >>> sorted(z)
        ['A', 'A-B', 'B', 'B serré', 'B-C', 'C serré', 'large']
        >>> z["large"][3] < 1.1 and z["B serré"][3] > z["B"][3] > z["A-B"][3]
        True

    C alone would show B (C is close to the camera, its shot is wide): there is only
    ``"B-C"`` and the tight shot of C; the tight shot of A would be almost ``"A"``.
    B alone is centred on B:

        >>> cam, cx, cy, zoom = z["B"]
        >>> cx
        950.0
    """
    names = list(people)
    ps = [people[n] for n in names]

    def label(ks):
        i, j = min(ks), max(ks)
        if i == 0 and j == len(ps) - 1:
            return "large"
        return names[i] if i == j else f"{names[i]}-{names[j]}"

    frames = []  # (shown, members, cx, cy, z)
    tights = {}
    for i in range(len(ps)):
        for j in range(i, len(ps)):
            members = list(range(i, j + 1))
            if main in names[i : j + 1]:
                m = names.index(main)
            else:
                m = min(members, key=lambda k: ps[k].top)
            cx, cy, z = _frame(ps, members, m, margin, eye_line, tolerance, max_zoom)
            # who is really in it (most of their box): near the edge of the picture, a person
            # alone may not be framed without a neighbour
            left, right = cx - W / z / 2, cx + W / z / 2
            shown = frozenset(
                k
                for k, p in enumerate(ps)
                if min(p.x1, right) - max(p.x0, left) > 0.6 * (p.x1 - p.x0)
            ) | frozenset(members)
            frames.append((shown, frozenset(members), cx, cy, z))
            if i == j and names[i] not in no_tight:
                # tight: zoom in, the person centred; the eyes on the third line, or higher
                # so as to keep the hands in; zoom in less if the head would be cut
                p = ps[i]
                zt = min(z * tight, max_zoom * tight)
                while True:
                    h = H / zt
                    y0 = min(max(p.eyes - eye_line * h, 0), H - h)
                    y0 = max(y0, min(p.bottom + 0.02 * h - h, H - h))  # the hands in
                    if y0 <= p.top - 0.02 * h or zt <= z:
                        break
                    zt *= 0.97
                x = min(max((p.x0 + p.x1) / 2, W / zt / 2), W - W / zt / 2)
                tights[f"{names[i]} serré"] = (
                    camera,
                    float(x),
                    float(y0 + h / 2),
                    float(zt),
                )
    # a framing is named after the people it shows; two framings showing the same people at
    # about the same zoom are one (the one computed for exactly those people is kept)
    out, kept = {}, {}  # kept: name -> (shown, members, zoom)

    def twin(shown, z):
        return next(
            (
                n
                for n, (s, _, zk) in kept.items()
                if s == shown and abs(zk / z - 1) < 0.15
            ),
            None,
        )

    for shown, members, cx, cy, z in sorted(frames, key=lambda f: len(f[1]) == 1):
        if len(members) == 1 and shown != members:
            # a person alone who cannot be framed without a neighbour
            if twin(shown, z):
                continue
            (k,) = members
            name = (
                f"{names[k]} (+ {', '.join(names[o] for o in sorted(shown - members))})"
            )
        else:
            name = label(shown)
            t = twin(shown, z)
            if t is not None:
                if kept[t][1] == shown or members != shown:
                    continue
                del out[t], kept[t]
        kept[name] = (shown, members, z)
        out[name] = (camera, float(cx), float(cy), float(z))
    # almost the same picture (near the edge of the picture, or a person close to the camera):
    # a person with a neighbour gives way to their tight shot; a tight shot that is almost a
    # group gives way to the group (a real tight shot is not possible there)
    for t, ft in list(tights.items()):
        for n in [n for n in out if "(+" in n]:
            if n.startswith(t.removesuffix(" serré")) and _overlap(ft, out[n]) > 0.75:
                del out[n]
        if any(_overlap(ft, f) > 0.75 for f in out.values()):
            warnings.warn(f"{t}: almost the same as a group framing, dropped")
            del tights[t]
    out.update(tights)
    return out


def _overlap(a, b):
    """Intersection over union of two framings."""
    ra = [
        a[1] - W / a[3] / 2,
        a[2] - H / a[3] / 2,
        a[1] + W / a[3] / 2,
        a[2] + H / a[3] / 2,
    ]
    rb = [
        b[1] - W / b[3] / 2,
        b[2] - H / b[3] / 2,
        b[1] + W / b[3] / 2,
        b[2] + H / b[3] / 2,
    ]
    iw = max(0.0, min(ra[2], rb[2]) - max(ra[0], rb[0]))
    ih = max(0.0, min(ra[3], rb[3]) - max(ra[1], rb[1]))
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(ra) + area(rb) - inter)


def framing_sheet(
    video: str | os.PathLike,
    time: float,
    framings: dict[str, tuple[str, float, float, float]],
    dst: str | os.PathLike,
    columns: int = 4,
    width: int = 480,
) -> Path:
    """Sheet of framings: each one cut from a picture of the video, with its name.

    Requires ffmpeg and OpenCV (extra ``concert``).

    Parameters
    ----------
    video : str or os.PathLike
        Source video (1920 x 1080).
    time : float
        Time of the picture in the video (s).
    framings : dict
        ``name -> (camera, centre_x, centre_y, zoom)``, e.g. from :func:`interval_framings`.
    dst : str or os.PathLike
        Sheet to write (.jpg).
    columns, width : int
        Layout of the sheet.
    """
    import cv2

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{time:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    img = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
    height = width * H // W
    tiles = []
    for name, (_, cx, cy, z) in framings.items():
        w, h = W / z, H / z
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        tile = cv2.resize(
            img[max(y0, 0) : y0 + int(h), max(x0, 0) : x0 + int(w)], (width, height)
        )
        label = name.replace("é", "e")  # OpenCV writes ASCII only
        cv2.putText(tile, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(
            tile, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack(
        [np.hstack(tiles[k : k + columns]) for k in range(0, len(tiles), columns)]
    )
    dst = Path(dst)
    cv2.imwrite(str(dst), sheet)
    return dst
