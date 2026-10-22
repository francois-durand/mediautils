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
    alone : float, optional
        Lowest thing to be seen in the shots of the person alone, if lower (the drums:
        the shots of the drummer show them, the groups need not).
    tight_hands : bool
        Whether the tight shot keeps the hands (an instrumentalist) or not (the singer:
        the head and the microphone).
    """

    x0: float
    x1: float
    top: float
    eyes: float
    bottom: float
    alone: float | None = None
    tight_hands: bool = True


def _frame(
    people, members, main, margin, eye_line, tolerance, max_zoom=1e9, group_eye_line=0.2
):
    """Frame (cx, cy, zoom) showing `members` (indices in `people`)."""
    ms = [people[i] for i in members]
    x0, x1 = min(p.x0 for p in ms), max(p.x1 for p in ms)
    top, bottom = min(p.top for p in ms), max(p.bottom for p in ms)
    if len(ms) == 1 and ms[0].alone is not None:
        bottom = max(bottom, ms[0].alone)
    eyes = people[main].eyes
    w = (x1 - x0) * (1 + 2 * margin)
    # the heads and the hands inside, with a margin; the eyes of a person alone on the third
    # line, or higher (up to a quarter) if that keeps the shot tighter; a group is framed
    # lower, its eyes at a fifth (the instruments are better seen: Fanfan), or lower in the
    # frame (up to the third) if a head needs more air: the tightest shot, then the lowest
    if len(members) == 1:
        lines = np.linspace(eye_line, min(eye_line, 0.25), 9)
    else:
        lines = np.linspace(min(eye_line, group_eye_line), eye_line, 9)
    needed = [
        max(w * H / W, (eyes - top) / (e - margin), (bottom - eyes) / (1 - e - margin))
        for e in lines
    ]
    k = next(k for k, he in enumerate(needed) if he <= min(needed) * 1.02)
    h, e = needed[k], lines[k]
    h = min(max(h, H / max_zoom), H)
    w = h * W / H
    y0 = min(max(eyes - e * h, 0), H - h)
    if y0 > top - margin * h:  # a head would be cut: raise the frame
        y0 = max(top - margin * h, 0)
    # horizontal position: its people whole, with room (they move: showing them whole wins
    # over hiding a neighbour, Fanfan), and the edges do not cut the others in the middle
    others = [p for k, p in enumerate(people) if k not in members]
    room = 1.5 * margin * w
    lo, hi = max(x1 + room - w, 0), min(x0 - room, W - w)
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
    tight: float = 1.22,
    tolerance: float = 0.15,
    no_tight: tuple[str, ...] = (),
    max_zoom: float = 2.4,
    fixed: dict[str, tuple[float, float, float]] | None = None,
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
    fixed : dict, optional
        Framings placed by eye, replacing the computed ones: ``name -> (centre_x,
        centre_y, zoom)``, the name as returned (``"Fanfan-Laetitia"``, ``"Thomas serré"``).
        They are taken as they are (Fanfan validates them on the pictures).
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
        ['A', 'A-B', 'B', 'B-C', 'C serré', 'large']
        >>> z["large"][3] < 1.1 and z["B"][3] > z["A-B"][3]
        True

    C alone would be almost ``"B-C"`` (C is close to the camera and to the edge, its
    shot is wide); the tight shots of A and B would not be clearly tighter than A and B
    alone (their hands must stay in). B alone is centred on B:

        >>> cam, cx, cy, zoom = z["B"]
        >>> cx
        950.0
    """
    items = _decide(
        people, camera, main, margin, eye_line, tight, tolerance, no_tight, max_zoom
    )
    return {
        name: f for _, name, f, problem in _fix(items, fixed, camera) if not problem
    }


def _fix(items, fixed, camera):
    """The framings placed by eye replace the computed ones (and are valid)."""
    fixed = fixed or {}
    unknown = set(fixed) - {name for _, name, _, _ in items}
    if unknown:
        raise ValueError(f"unknown framings: {sorted(unknown)}")
    return [
        (label, name, (camera, *map(float, fixed[name])), "")
        if name in fixed
        else (label, name, f, p)
        for label, name, f, p in items
    ]


def _decide(
    people, camera, main, margin, eye_line, tight, tolerance, no_tight, max_zoom
):
    """Every framing, in the order of the axis: ``(file label, name, framing, problem)``.

    For each person, the intervals that start with them from the largest, then the person
    alone and their tight shot. `problem` is ``""`` for a framing that works, else why not:

    - a group that shows other people too, or that is almost a larger group;
    - a person alone (who may show their neighbours) that is almost a group;
    - a tight shot that is almost a group, or not clearly tighter than the person alone
      (then, if the shot of the person alone shows other people, the tight shot wins).
    """
    names = list(people)
    n = len(names)
    frames, tights = _candidates(
        people, camera, main, margin, eye_line, tight, tolerance, no_tight, max_zoom
    )
    by_members = {f[1]: f for f in frames}

    def framing(members):
        _, _, cx, cy, z = by_members[frozenset(members)]
        return (camera, float(cx), float(cy), float(z))

    def name(i, j):
        if i == 0 and j == n - 1:
            return "large"
        return names[i] if i == j else f"{names[i]}-{names[j]}"

    problems = {}
    # groups: they show their people only; of two almost equal groups, the larger is kept
    groups = sorted(
        ((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda g: g[0] - g[1]
    )
    valid = {}
    for i, j in groups:
        shown, members = by_members[frozenset(range(i, j + 1))][:2]
        if shown != members:
            problems[i, j] = "montre aussi " + ", ".join(
                names[k] for k in sorted(shown - members)
            )
            continue
        f = framing(range(i, j + 1))
        twin = next((g for g in valid if _overlap(f, valid[g]) > 0.85), None)
        if twin:
            problems[i, j] = f"presque le meme que {name(*twin)}"
            continue
        valid[i, j] = f
    # people alone: centred on them, their neighbours may be seen
    for i in range(n):
        f = framing([i])
        # (0.8: a person close to the camera at the edge, framed with their hands, is a group)
        twin = next((g for g in valid if _overlap(f, valid[g]) > 0.8), None)
        if twin:
            problems[i, i] = f"presque le meme que {name(*twin)}"
    # tight shots
    for i, ft in tights.items():
        twin = next((g for g in valid if _overlap(ft, valid[g]) > 0.75), None)
        if twin:
            problems[i, "t"] = f"presque le meme que {name(*twin)}"
        elif (i, i) not in problems and ft[3] < 1.15 * framing([i])[3]:
            shown, members = by_members[frozenset([i])][:2]
            if shown != members:
                problems[i, i] = f"presque le meme que {names[i]} serré"
            else:
                problems[i, "t"] = f"pas vraiment plus serre que {names[i]}"
    items = []
    for i in range(n):
        for j in range(n - 1, i - 1, -1):
            label = names[i] if i == j else f"{names[i]}_{names[j]}"
            items.append(
                (label, name(i, j), framing(range(i, j + 1)), problems.get((i, j), ""))
            )
        if i in tights:
            items.append(
                (
                    f"{names[i]}_serré",
                    f"{names[i]} serré",
                    tights[i],
                    problems.get((i, "t"), ""),
                )
            )
    return items


def _candidates(
    people, camera, main, margin, eye_line, tight, tolerance, no_tight, max_zoom
):
    """Every interval framing (shown, members, cx, cy, z) and tight shot {i: framing}."""
    names = list(people)
    ps = [people[n] for n in names]
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
                # tight: zoom in, the person centred; the eyes a little above the third line
                # (Fanfan), or higher so as to keep the hands in; zoom in less if the head
                # would be cut
                p = ps[i]
                zt = min(z * tight, max_zoom * tight)
                while True:
                    h = H / zt
                    # the singer's eyes higher still: less room above the head (Fanfan)
                    e = min(eye_line, 0.28 if p.tight_hands else 0.24)
                    y0 = min(max(p.eyes - e * h, 0), H - h)
                    low = p.bottom if p.alone is None else max(p.bottom, p.alone)
                    if not p.tight_hands:  # a singer: the tight shot is the head
                        low = p.eyes
                    y0 = max(y0, min(low + 0.02 * h - h, H - h))  # the hands in
                    # some air above the head (Fanfan: the edge must not scratch it)
                    if y0 <= p.top - 0.08 * h or zt <= z:
                        break
                    zt *= 0.97
                x = min(max((p.x0 + p.x1) / 2, W / zt / 2), W - W / zt / 2)
                tights[i] = (camera, float(x), float(y0 + h / 2), float(zt))
    return frames, tights


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


def framing_files(
    video: str | os.PathLike,
    time: float,
    people: dict[str, Person],
    dst: str | os.PathLike,
    side: str = "cour",
    label_time: float | None = None,
    width: int = 1920,
    picture: np.ndarray | None = None,
    prefix: str | None = None,
    time_last: bool = False,
    **kwargs,
) -> list[Path]:
    """One picture per framing, every interval and tight shot, to be checked one by one.

    The files are named ``t0100_cour_01_Thomas_Mathieu.jpg`` (time of the picture,
    side of the camera, number, first and last person of the interval, or
    ``Thomas_serré``), in the order of the axis: for each person, the intervals that
    start with them from the largest, then the person alone and their tight shot. A
    framing that does not work (it shows other people too, or it is almost another
    one) is drawn anyway, crossed in red, with the reason.

    Requires ffmpeg and OpenCV (extra ``concert``).

    Parameters
    ----------
    video : str or os.PathLike
        Source video (1920 x 1080).
    time : float
        Time of the picture in the video (s).
    people : dict[str, Person]
        As for :func:`interval_framings`.
    dst : str or os.PathLike
        Folder to write to (created if needed).
    side : str
        Side of the camera, for the names (``"cour"``, ``"jardin"``).
    label_time : float, optional
        Time written in the names (s), e.g. the time in the project rather than in the
        video; default `time`.
    width : int
        Width of the pictures.
    picture : np.ndarray, optional
        Picture to cut the framings from (BGR, 1920 x 1080) instead of the video at
        `time`, e.g. from :func:`stacked_exposure`.
    prefix : str, optional
        Start of the names instead of the time (e.g. ``"empilement"``).
    time_last : bool
        Names ``cour_01_Thomas_Mathieu_t0100.jpg``: with several times, the pictures of
        a framing follow one another (Fanfan: 5 times, easier to judge).
    **kwargs
        Options of :func:`interval_framings` (``fixed`` included).

    Returns
    -------
    list[Path]
        The pictures written.
    """
    import unicodedata

    import cv2

    camera = kwargs.pop("camera", "L")
    opts = dict(
        main=None,
        margin=0.04,
        eye_line=1 / 3,
        tight=1.22,  # Fanfan: the tight shot of the singer a little looser (his elbow in)
        tolerance=0.15,
        no_tight=(),
        max_zoom=2.4,
    )
    fixed = kwargs.pop("fixed", None)
    opts.update(kwargs)
    items = [
        (label, f, problem)
        for label, _, f, problem in _fix(_decide(people, camera, **opts), fixed, camera)
    ]
    img = _picture(video, time) if picture is None else picture
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t = round(time if label_time is None else label_time)
    when = prefix or f"t{t // 60:02d}{t % 60:02d}"
    stem = f"{when}_{side}"
    out = []
    for k, (label, (_, cx, cy, z), problem) in enumerate(items, 1):
        w, h = W / z, H / z
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        pic = cv2.resize(
            img[max(y0, 0) : y0 + int(h), max(x0, 0) : x0 + int(w)],
            (width, width * H // W),
        )

        def ascii_(s):  # OpenCV writes ASCII only
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

        text = (
            f"{label.replace('_serré', ' serré').replace('_', ' + ')}  (zoom {z:.2f})"
        )
        s = width / 960  # sizes of the text and of the cross follow the picture

        def write(t, y, colour):
            for c, th in (((0, 0, 0), 5), (colour, 2)):
                cv2.putText(
                    pic,
                    ascii_(t),
                    (int(10 * s), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    s,
                    c,
                    max(1, int(th * s)),
                )

        write(text, 36 * s, (0, 255, 255))
        if problem:
            hh = pic.shape[0]
            cv2.line(pic, (0, 0), (width - 1, hh - 1), (0, 0, 255), int(8 * s))
            cv2.line(pic, (0, hh - 1), (width - 1, 0), (0, 0, 255), int(8 * s))
            write(problem, hh - 16 * s, (0, 0, 255))
        path = dst / (
            f"{side}_{k:02d}_{label}_{when}.jpg"
            if time_last
            else f"{stem}_{k:02d}_{label}.jpg"
        )
        ok, buf = cv2.imencode(
            ".jpg", pic
        )  # imwrite cannot write non-ASCII paths on Windows
        path.write_bytes(buf.tobytes())
        out.append(path)
    return out


def long_exposure(
    video: str | os.PathLike,
    start: float,
    end: float,
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Long exposure of a part of a video: where people stand, and where they go.

    One picture every `step` seconds between `start` and `end` (s of the video).
    Requires ffmpeg.

    Returns
    -------
    mean : np.ndarray
        Average picture (BGR, uint8): where somebody stands most of the time, they are
        almost sharp; where they go now and then, a pale ghost.
    presence : np.ndarray
        The still background (median picture), and on it the people, opaque wherever
        they go a little (a quarter of the time is enough): the average of the pictures
        only where they differ clearly from the background (BGR, uint8).
    """
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video),
            "-vf",
            f"fps={1 / step},scale={W}:{H}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    n = len(raw) // (W * H * 3)
    frames = np.frombuffer(raw[: n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    s = np.zeros((H, W, 3))
    for f in frames:  # in float, one picture at a time (memory)
        s += f
    mean = s / n
    # presence, at half resolution (the median of every picture is heavy)
    half = frames[:, ::2, ::2]
    background = np.empty(half.shape[1:], np.uint8)
    for r in range(0, half.shape[1], 30):  # by bands of rows (memory)
        background[r : r + 30] = np.median(half[:, r : r + 30], axis=0)
    bg = background.astype(np.int16)
    fg_sum = np.zeros(half.shape[1:])
    count = np.zeros(half.shape[1:3])
    for f in half:
        moving = np.abs(f.astype(np.int16) - bg).sum(axis=2) > 45
        fg_sum[moving] += f[moving]
        count += moving
    fg = fg_sum / np.maximum(count, 1)[..., None]
    alpha = (np.clip(count / n * 4, 0, 1) ** 0.7)[..., None]
    presence = background * (1 - alpha) + fg * alpha
    import cv2

    presence = cv2.resize(np.clip(presence, 0, 255).astype(np.uint8), (W, H))
    return mean.astype(np.uint8), presence


def stacked_exposure(
    video: str | os.PathLike,
    start: float,
    end: float,
    step: float = 1.0,
    tolerance: float = 45,
    moments: int | None = None,
) -> np.ndarray:
    """Every position of the people of a part of a video, sharp, on one picture.

    With `moments`, only that many pictures, evenly spread, are stacked (the median
    still comes from every picture): easier to read (Fanfan: 10).

    One picture every `step` seconds; each one but the first is transparent where it is
    the same as the median picture (within `tolerance`, sum of the three channels
    0-255), and they are stacked in order, the first one behind: a chronophotography
    (Fanfan's idea). Where somebody stands most of the time, the median is them, so the
    rare pictures where they are away may leave holes in them.

    Requires ffmpeg and OpenCV (extra ``concert``). Computed at half resolution.

    Returns
    -------
    np.ndarray
        The picture (BGR, uint8, 1920 x 1080).
    """
    import cv2

    w, h = W // 2, H // 2
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video),
            "-vf",
            f"fps={1 / step},scale={w}:{h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    n = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    median = np.empty((h, w, 3), np.uint8)
    for r in range(0, h, 30):  # by bands of rows (memory)
        median[r : r + 30] = np.median(frames[:, r : r + 30], axis=0)
    med = median.astype(np.int16)
    chosen = (
        frames
        if moments is None
        else frames[np.linspace(0, n - 1, moments).round().astype(int)]
    )
    stack = chosen[0].copy()
    for f in chosen[1:]:
        moving = np.abs(f.astype(np.int16) - med).sum(axis=2) > tolerance
        stack[moving] = f[moving]
    return cv2.resize(stack, (W, H))


def _picture(video, time):
    """Picture of a video at a time (BGR, 1920 x 1080)."""
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
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)
