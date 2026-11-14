"""Revision of a finished montage, and the checks before publishing it.

Made for the revision of Burlingue's concert of June 2026 (2026-11), when validated
montages had to be changed in a few places (children's faces in the audience, retouches
asked on the export) and exported again. This module

- reads the markers of a Shotcut project (:func:`read_markers`, :func:`marker_time`);
- writes a copy of a project where passages take another framing, or a cut moves,
  without touching the rest (:func:`edit_frames`, :func:`move_cut`);
- shows what is in the rush and what the montage shows, read from the source files with
  ffmpeg, much faster than a render (:func:`rush_sheets`, :func:`montage_views`,
  :func:`ruler_image`);
- compares a new export with the validated one (:func:`compare_exports`);
- runs a queue of exports, one at a time (:func:`run_export_queue`).
"""

import copy
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mediautils.audit import _crop, _visible
from mediautils.montage import _clock_of, _prop, _write


# ----- markers


def read_markers(project: str | os.PathLike) -> list[tuple[str, float]]:
    """Markers of a Shotcut project: ``(text, time in s)``, in time order.

    Parameters
    ----------
    project : str | os.PathLike
        The project.

    Returns
    -------
    list[tuple[str, float]]
        The markers.
    """
    root = ET.parse(project).getroot()
    C = _clock_of(root)
    out = []
    for box in root.iter("properties"):
        if box.get("name") != "shotcut:markers":
            continue
        for m in box:
            text, start = _prop(m, "text"), _prop(m, "start")
            if start is not None:
                out.append((text or "", C.frames(start) / C.fps))
    return sorted(out, key=lambda x: x[1])


def marker_time(project: str | os.PathLike, label: str) -> float:
    """Time (s) of a marker, or of a Shotcut timecode.

    Parameters
    ----------
    project : str | os.PathLike
        The project.
    label : str
        A bar marker (``"15.1"``; ``"15"`` means ``"15.1"``; a marker ``"15.1 Couplet"``
        matches), or a Shotcut timecode ``"HH:MM:SS:FF"`` (frames at the project's frame
        rate, counted from 0).

    Returns
    -------
    float
        The time.

    Raises
    ------
    KeyError
        No such marker.
    """
    if label.count(":") == 3:
        fps = _clock_of(ET.parse(project).getroot()).fps
        h, m, s, f = (int(x) for x in label.split(":"))
        return h * 3600 + m * 60 + s + f / fps
    marks = read_markers(project)
    wanted = {label, label + ".1"} if "." not in label else {label}
    for text, t in marks:
        if text.split(" ")[0] in wanted:
            return t
    if label.endswith(".1"):  # the 1 of a bar may be marked "15" or "15 Couplet"
        for text, t in marks:
            if text.split(" ")[0] == label[:-2]:
                return t
    raise KeyError(f"{label}: no such marker")


# ----- small edits of a project


def _playlist(root, track):
    for p in root.iter("playlist"):
        if _prop(p, "shotcut:name") == track:
            return p
    raise KeyError(f"no track named {track}")


def _affine(chain):
    return next(
        (f for f in chain.iter("filter") if _prop(f, "mlt_service") == "affine"), None
    )


def edit_frames(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    ops: list[tuple[float, float, str | tuple[float, float, float, float], str | None]],
    track: str = "V1",
) -> Path:
    """Write a copy of a montage where passages of a track take another framing.

    The shots are cut at the bounds of each passage; inside it, each piece keeps its
    source and its in-point (so pictures and sound stay in sync) and only its framing (the
    Shotcut "Size, Position & Rotate" filter) changes. The rest of the montage is kept as
    it is, moving zooms and cross-fades included. Neighbouring pieces with the same new
    framing and a continuous source become one shot.

    Parameters
    ----------
    src : str | os.PathLike
        The montage (not modified).
    dst : str | os.PathLike
        The new montage (must not exist).
    ops : list of (start, end, frame, name)
        Passages, in s of the timeline (rounded to the frame). `frame` is the id of a
        chain of the montage whose fixed framing is copied, or a rectangle
        ``(x0, y0, x1, y1)`` in pixels of the project's profile, 16:9. `name` replaces the
        framing's name (``montage:shot``); ``None`` keeps the model's.
    track : str
        Name of the video track.

    Returns
    -------
    Path
        `dst`.

    Raises
    ------
    FileExistsError
        `dst` exists.
    ValueError
        A passage covers a gap of the track, a model is a moving framing, a rectangle is
        not 16:9, or no fixed framing is there to serve as a template for a rectangle.
    """
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    C = _clock_of(root)
    prof = root.find("profile")
    W, H = int(prof.get("width")), int(prof.get("height"))

    def rect_text(box):
        x0, y0, x1, y1 = box
        z = W / (x1 - x0)
        if abs((y1 - y0) * z - H) >= 2:
            raise ValueError(f"{box}: not the aspect ratio of the picture")
        return f"{-x0 * z:.1f} {-y0 * z:.1f} {W * z:.1f} {H * z:.1f} 1"

    chains = {c.get("id"): c for c in root if c.tag in ("chain", "producer")}
    template = next(
        (
            _affine(c)
            for c in chains.values()
            if _affine(c) is not None
            and "=" not in (_prop(_affine(c), "transition.rect") or "=")
        ),
        None,
    )
    pl = _playlist(root, track)
    items = [e for e in pl if e.tag in ("entry", "blank")]
    # pieces: [start, end (excluded), in-point, chain id, replacement]
    pieces, pos = [], 0
    for e in items:
        if e.tag == "blank":
            n = C.frames(e.get("length"))
            pieces.append([pos, pos + n, None, None, None])
        else:
            n = C.frames(e.get("out")) - C.frames(e.get("in")) + 1
            pieces.append(
                [pos, pos + n, C.frames(e.get("in")), e.get("producer"), None]
            )
        pos += n
    total = pos
    for k, (t0, t1, frame, name) in enumerate(ops):
        f0, f1 = round(t0 * C.fps), round(t1 * C.fps)
        for cut in (f0, f1):
            for i, p in enumerate(pieces):
                if p[0] < cut < p[1]:
                    q = list(p)
                    p[1] = q[0] = cut
                    if q[2] is not None:
                        q[2] = p[2] + (cut - p[0])
                    pieces.insert(i + 1, q)
                    break
        for p in pieces:
            if f0 <= p[0] and p[1] <= f1:
                if p[3] is None:
                    raise ValueError(f"{t0}-{t1}: a gap of the track")
                p[4] = (k, frame, name)
    for e in items:
        pl.remove(e)

    def resource(cid):
        return _prop(chains[cid], "resource")

    out = []
    for p in pieces:  # same replacement, continuous source: one shot
        prev = out[-1] if out else None
        if (
            prev
            and p[4] is not None
            and prev[4] is not None
            and prev[4][1:] == p[4][1:]
            and prev[2] + (prev[1] - prev[0]) == p[2]
            and resource(prev[3]) == resource(p[3])
        ):
            prev[1] = p[1]
        else:
            out.append(list(p))
    used = set()
    for j, (a, b, i0, cid, rep) in enumerate(out):
        if cid is None:
            ET.SubElement(pl, "blank", length=C.string(b - a))
            continue
        i1 = i0 + (b - a) - 1
        if rep is None and cid not in used:
            prod = cid
        else:
            c = copy.deepcopy(chains[cid])
            prod = f"{cid}_e{j}"
            c.set("id", prod)
            for f in c.iter("filter"):
                if f.get("id"):
                    f.set("id", f"{f.get('id')}_e{j}")
            if rep is not None:
                _, frame, name = rep
                old = _affine(c)
                if isinstance(frame, str):
                    new = copy.deepcopy(_affine(chains[frame]))
                    if "=" in (_prop(new, "transition.rect") or ""):
                        raise ValueError(f"{frame}: a moving framing")
                    if name is None:
                        name = _prop(chains[frame], "montage:shot")
                else:
                    if template is None:
                        raise ValueError("no fixed framing in the montage to copy")
                    new = copy.deepcopy(template)
                    new.find("property[@name='transition.rect']").text = rect_text(
                        frame
                    )
                new.set("id", f"revise_affine_e{j}")
                new.set("in", C.string(i0))
                new.set("out", C.string(i1))
                if old is not None:
                    c.insert(list(c).index(old), new)
                    c.remove(old)
                else:
                    c.append(new)
                if name is not None:
                    pn = c.find("property[@name='montage:shot']")
                    if pn is None:
                        pn = ET.Element("property", name="montage:shot")
                        c.insert(0, pn)
                    pn.text = name
            root.insert(list(root).index(chains[cid]) + 1, c)
        used.add(cid)
        ET.SubElement(
            pl, "entry", producer=prod, **{"in": C.string(i0), "out": C.string(i1)}
        )
    assert sum(p[1] - p[0] for p in out) == total, "length changed"
    _write(tree, dst)
    return dst


def move_cut(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    old: float,
    new: float,
    track: str = "V1",
) -> Path:
    """Write a copy of a montage where a cut between two shots moves.

    The shot before ends earlier (or later), the shot after starts as much earlier (or
    later) in its source; the filters of the two shots (framings) follow their in and
    out points. Nothing else moves.

    Parameters
    ----------
    src : str | os.PathLike
        The montage (not modified).
    dst : str | os.PathLike
        The new montage (must not exist).
    old, new : float
        The cut, before and after, in s of the timeline.
    track : str
        Name of the video track.

    Returns
    -------
    Path
        `dst`.

    Raises
    ------
    FileExistsError
        `dst` exists.
    ValueError
        No cut between two shots at `old`, or a shot that would vanish.
    """
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    C = _clock_of(root)
    pl = _playlist(root, track)
    entries = [e for e in pl if e.tag in ("entry", "blank")]
    pos, k = 0, None
    for i, e in enumerate(entries[:-1]):
        if e.tag == "blank":
            pos += C.frames(e.get("length"))
        else:
            pos += C.frames(e.get("out")) - C.frames(e.get("in")) + 1
        if pos == round(old * C.fps):
            k = i
            break
    if k is None or not entries[k].tag == entries[k + 1].tag == "entry":
        raise ValueError(f"{old}: not a cut between two shots")
    d = round(new * C.fps) - round(old * C.fps)
    a, b = entries[k], entries[k + 1]
    chains = {c.get("id"): c for c in root if c.tag in ("chain", "producer")}
    ids = [x.get("producer") for x in entries if x.tag == "entry"]
    for e, attr in ((a, "out"), (b, "in")):
        cid = e.get("producer")
        if (
            ids.count(cid) != 1
        ):  # a chain shared by several shots: this shot gets its own copy
            c = copy.deepcopy(chains[cid])
            new_id = f"{cid}_cut{k}{attr}"
            c.set("id", new_id)
            for f in c.iter("filter"):
                if f.get("id"):
                    f.set("id", f"{f.get('id')}_cut{k}{attr}")
            root.insert(list(root).index(chains[cid]) + 1, c)
            chains[new_id] = c
            e.set("producer", new_id)
        before = e.get(attr)
        e.set(attr, C.string(C.frames(before) + d))
        for f in chains[e.get("producer")].iter("filter"):
            if f.get(attr) == before:
                f.set(attr, e.get(attr))
    for e in (a, b):
        if C.frames(e.get("out")) < C.frames(e.get("in")):
            raise ValueError(f"{new}: a shot would vanish")
    _write(tree, dst)
    return dst


# ----- pictures


def _grab(project, t, C, pieces, width, height, dst, crop=True):
    """Picture of the montage at `t` s, read from the source of the visible clip; False if black."""
    n = int(round(t * C.fps))
    hit = next((p for p in pieces if p[0] <= n < p[1]), None)
    if hit is None or hit[3] is None:
        return False
    _, _, _, chain, i0, pos = hit
    k = n - pos + i0
    vf = f"{_crop(chain, k, C)}," if crop else ""
    fit = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{k / C.fps:.3f}",
            "-i",
            str(project.parent / _prop(chain, "resource")),
            "-frames:v",
            "1",
            "-vf",
            vf + fit,
            str(dst),
        ],
        check=True,
    )
    return True


def _sheet(project, times, dst, cols, width, crop):
    from PIL import Image, ImageDraw, ImageFont

    project = Path(project).resolve()
    C, pieces = _visible(project)
    height = width * 9 // 16
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, max(rows, 1) * height))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(14, width // 25))
    tmp = Path(dst).with_suffix(".tmp.png")
    for j, t in enumerate(times):
        if not _grab(project, t, C, pieces, width, height, tmp, crop):
            continue
        x, y = (j % cols) * width, (j // cols) * height
        sheet.paste(Image.open(tmp).convert("RGB"), (x, y))
        draw.text(
            (x + 8, y + 6),
            f"{int(t // 60)}:{t % 60:04.1f}",
            fill="yellow",
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )
    tmp.unlink(missing_ok=True)
    sheet.save(dst, quality=85)
    return Path(dst)


def montage_views(
    project: str | os.PathLike,
    times: list[float],
    dst: str | os.PathLike,
    cols: int = 4,
    width: int = 960,
) -> Path:
    """What the montage shows at given times, on one sheet.

    Each picture is read from the source of the top visible clip and cropped as its
    framing (moving zooms interpolated), with ffmpeg: much faster than a render, and
    right for checking a framing (e.g. that a child's face stays out of it). Pictures
    (logos) over the cameras are left out.

    Parameters
    ----------
    project : str | os.PathLike
        The montage.
    times : list[float]
        Times, in s of the timeline.
    dst : str | os.PathLike
        The sheet (JPEG).
    cols : int
        Pictures per row.
    width : int
        Width of a picture.

    Returns
    -------
    Path
        `dst`.
    """
    return _sheet(project, times, dst, cols, width, True)


def rush_sheets(
    project: str | os.PathLike,
    dst: str | os.PathLike,
    step: float = 5.0,
    cols: int = 4,
    rows: int = 4,
    width: int = 480,
) -> list[Path]:
    """The whole picture of the rush under the montage, every `step` s, on sheets.

    A framing only shows part of the rush: looking at the whole picture shows at least all
    the montage can show. Made to look for faces that must not be seen (children of the
    audience) before a closer look with :func:`montage_views`.

    Parameters
    ----------
    project : str | os.PathLike
        The montage.
    dst : str | os.PathLike
        Folder of the sheets (created): ``sheet_01.jpg``, ...
    step : float
        Time between two pictures, in s.
    cols, rows : int
        Pictures per row, rows per sheet.
    width : int
        Width of a picture.

    Returns
    -------
    list[Path]
        The sheets.
    """
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    C, pieces = _visible(Path(project).resolve())
    end = max((p[1] for p in pieces if p[3] is not None), default=0) / C.fps
    times = list(np.arange(0, end, step))
    n = cols * rows
    return [
        _sheet(
            project,
            times[k : k + n],
            dst / f"sheet_{k // n + 1:02d}.jpg",
            cols,
            width,
            False,
        )
        for k in range(0, len(times), n)
    ]


def ruler_image(
    project: str | os.PathLike,
    t: float,
    dst: str | os.PathLike,
    boxes: list[tuple[float, float, float, float]] = (),
    width: int = 1920,
) -> Path:
    """The whole picture of the rush at a time of the montage, with a pixel ruler and frames drawn.

    To place a framing by eye, to the pixel: the ruler counts pixels of the project's
    profile (the unit of the framings), a tick every 100, a number every 200; `boxes` are
    drawn in yellow, cyan, magenta, lime, orange.

    Parameters
    ----------
    project : str | os.PathLike
        The montage.
    t : float
        Time, in s of the timeline.
    dst : str | os.PathLike
        The picture (JPEG).
    boxes : list of (x0, y0, x1, y1)
        Frames to draw, in pixels of the profile.
    width : int
        Width of the picture.

    Returns
    -------
    Path
        `dst`.
    """
    from PIL import Image, ImageDraw, ImageFont

    project = Path(project).resolve()
    C, pieces = _visible(project)
    W, H = C.size
    height = round(width * H / W)
    if not _grab(project, t, C, pieces, width, height, dst, crop=False):
        Image.new("RGB", (width, height)).save(dst)
    im = Image.open(dst).convert("RGB")
    d = ImageDraw.Draw(im)
    k = width / W
    font = ImageFont.load_default(size=22)
    for v in range(100, W, 100):
        d.line([(v * k, 0), (v * k, 14 if v % 200 else 26)], fill="white", width=1)
        if v % 200 == 0:
            d.text(
                (v * k + 2, 26),
                str(v),
                fill="white",
                font=font,
                stroke_width=2,
                stroke_fill="black",
            )
    for v in range(100, H, 100):
        d.line([(0, v * k), (14 if v % 200 else 26, v * k)], fill="white", width=1)
        if v % 200 == 0:
            d.text(
                (28, v * k - 11),
                str(v),
                fill="white",
                font=font,
                stroke_width=2,
                stroke_fill="black",
            )
    for col, (x0, y0, x1, y1) in zip(
        ["yellow", "cyan", "magenta", "lime", "orange"], boxes
    ):
        d.rectangle([x0 * k, y0 * k, x1 * k - 1, y1 * k - 1], outline=col, width=3)
    im.save(dst, quality=88)
    return Path(dst)


# ----- exports


def _duration(f):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(f),
        ],
        capture_output=True,
        text=True,
    ).stdout
    return float(out)


def _level(f, t, d):
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-t",
            str(d),
            "-i",
            str(f),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
    ).stdout
    x = np.frombuffer(raw, np.float32)
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)) if len(x) else -np.inf


def _gray(f, t, w, h):
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(f),
            "-frames:v",
            "1",
            "-vf",
            f"scale={w}:{h},format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w).astype(float)


def compare_exports(
    new: str | os.PathLike,
    old: str | os.PathLike,
    t: float | None = None,
    corner: tuple[float, float] = (0.15, 0.23),
    sound: float = 20.0,
) -> dict[str, float]:
    """Compare a new export with the validated one, at one time.

    For a revision that should change nothing but a few passages (or add a logo): same
    duration, same sound level, same picture outside the bottom right corner (where the
    logo goes). Check at several times, among them just after the passages changed (if
    anything shifted, the pictures differ from there on), and look at the pictures of the
    passages changed in the new file itself.

    Parameters
    ----------
    new, old : str | os.PathLike
        The two exports.
    t : float, optional
        Time, in s (default: the middle of `new`).
    corner : tuple[float, float]
        Width and height of the corner, as fractions of the picture.
    sound : float
        Length of the sound measured from `t`, in s.

    Returns
    -------
    dict[str, float]
        ``duration_new``, ``duration_old`` (s), ``level_new``, ``level_old`` (dB),
        ``diff_outside``, ``diff_corner`` (mean absolute difference of grey levels, 0-255,
        on a 384x216 picture). An unchanged passage gives ``diff_outside`` below ~3; a
        logo added gives ``diff_corner`` above ~8.
    """
    dn, do = _duration(new), _duration(old)
    t = dn / 2 if t is None else t
    w, h = 384, 216
    a, b = _gray(new, t, w, h), _gray(old, t, w, h)
    mask = np.zeros_like(a, bool)
    mask[h - round(corner[1] * h) :, w - round(corner[0] * w) :] = True
    diff = np.abs(a - b)
    return {
        "duration_new": dn,
        "duration_old": do,
        "level_new": _level(new, t, sound),
        "level_old": _level(old, t, sound),
        "diff_outside": float(diff[~mask].mean()),
        "diff_corner": float(diff[mask].mean()),
    }


def _melt_running():
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq melt.exe"],
                capture_output=True,
                text=True,
            ).stdout
            return "melt.exe" in out
        return (
            subprocess.run(["pgrep", "-x", "melt"], capture_output=True).returncode == 0
        )
    except FileNotFoundError:
        return False


def run_export_queue(
    queue: str | os.PathLike,
    poll: float = 60.0,
    idle: float = 12 * 3600,
) -> None:
    """Run the commands of a queue file, one at a time, waiting while melt runs.

    Exports are long (about an hour for a 4K song) and must not run together: a queue file
    holds one command per line, run from the folder of the file (a line starting with a
    ``.py`` file runs with this Python). A line done is prefixed with ``fait`` (or
    ``ÉCHEC``); lines starting with ``#`` are skipped; a line ``FIN`` stops the queue. The
    file is read again before every command: to add one, write a line after the one
    running (never change the lines above it). Each command's output goes to
    ``export_<command>.log`` next to the queue; the start and end of each command are
    printed (redirect them to a log). Run it at low priority, the exports inherit it
    (e.g. ``start /belownormal /affinity F000`` on Windows). If the computer restarts,
    the export running is cut ("moov atom not found"): delete it, put its line back
    and start the queue again.

    Parameters
    ----------
    queue : str | os.PathLike
        The queue file.
    poll : float
        Time between two looks at the queue or at melt, in s.
    idle : float
        Stop after this time without anything to run, in s.
    """
    queue = Path(queue)
    here = queue.parent
    waited = 0.0
    while True:
        lines = queue.read_text(encoding="utf-8").splitlines()
        todo = [
            i
            for i, x in enumerate(lines)
            if x.strip() and not x.startswith(("fait ", "ÉCHEC ", "#"))
        ]
        if not todo:
            if waited >= idle:
                return
            time.sleep(poll)
            waited += poll
            continue
        i = todo[0]
        cmd = lines[i].strip()
        if cmd == "FIN":
            return
        while _melt_running():
            time.sleep(poll)
        print(time.strftime("%H:%M"), "export :", cmd, flush=True)
        args = cmd.split()
        if args[0].endswith(".py"):
            args = [sys.executable, *args]
        log = here / ("export_" + cmd.replace(" ", "_").replace(".py", "") + ".log")
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(args, cwd=here, stdout=f, stderr=subprocess.STDOUT)
        lines = queue.read_text(encoding="utf-8").splitlines()
        lines[i] = ("fait " if r.returncode == 0 else "ÉCHEC ") + cmd
        queue.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(time.strftime("%H:%M"), "fini :", cmd, r.returncode, flush=True)
        waited = 0.0
