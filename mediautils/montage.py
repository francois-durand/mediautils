"""Multi-camera montage of a concert song in Shotcut, from a synchronization project.

The starting point is a Shotcut project (``.mlt``) where the videos of the cameras
are synchronized, one per video track, and the sound is mixed on audio tracks.
One camera is the *reference* (``"L"``, matched by `reference` — the Lumix by
default); other sources are recognized by their file names (`others`, by default
``"W"`` for WhatsApp videos), and any other file is ``"A"`` (Andrea's camera).
From it, this module:

- reads where each source file sits on the timeline (:class:`SyncProject`);
- corrects the synchronization to the frame from the soundtracks
  (:func:`resync_project`);
- extracts one image per second and contact sheets to look at the song
  (:func:`extract_frames`, :func:`contact_sheets`), and its beats
  (:func:`timeline_beats`);
- generates a montage: a list of shots, each a camera or a fixed zoom on a
  musician, cut on the beats (:func:`nervous_shots`, :func:`build_montage`);
- cuts the head or the end of a project (:func:`trim_project_head`,
  :func:`trim_project_end`) and replaces its audio tracks by a single sound
  file (:func:`use_single_audio`), e.g. made by :mod:`mediautils.livesound`.

Generated projects are new files; the projects given as input are never modified
(except by the ``trim_*`` functions, documented as in place, meant for generated
files). Paths of source files inside the projects are kept relative.
"""

import copy
import os
import re
import subprocess
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

LUMIX = r"_P\d{7}\.MP4$"
OTHERS = {
    "W": r"-WA\d{4}"
}  # extra sources: label -> file name pattern (WhatsApp videos)


class MltClock:
    """Conversion between MLT clock strings (``"HH:MM:SS.mmm"``) and frames.

    Parameters
    ----------
    fps : float
        Frame rate of the project.

    Examples
    --------
        >>> clock = MltClock(60000 / 1001)
        >>> clock.frames("00:00:25.042")
        1501
        >>> clock.string(1501)
        '00:00:25.042'
        >>> clock.string(clock.frames("00:04:29.630"))
        '00:04:29.630'
    """

    def __init__(self, fps: float):
        self.fps = fps

    def frames(self, s: str) -> int:
        """Number of frames of a clock string."""
        h, m, sec = s.split(":")
        return round(self.fps * (int(h) * 3600 + int(m) * 60 + float(sec)))

    def string(self, frames: int) -> str:
        """Clock string of a number of frames (as MLT writes it)."""
        hours = int(frames / (self.fps * 3600))
        frames -= round(hours * 3600 * self.fps)
        mins = int(frames / (self.fps * 60))
        frames -= round(mins * 60 * self.fps)
        return f"{hours:02d}:{mins:02d}:{frames / self.fps:06.3f}"


def _prop(el, name):
    p = el.find(f"property[@name='{name}']")
    return None if p is None else p.text


def _write(tree, path):
    with open(path, "wb") as fh:
        fh.write(b'<?xml version="1.0" standalone="no"?>\n')
        tree.write(fh, encoding="utf-8", xml_declaration=False)


def _clock_of(root):
    prof = root.find("profile")
    return MltClock(int(prof.get("frame_rate_num")) / int(prof.get("frame_rate_den")))


class SyncProject:
    """A Shotcut project with one camera per video track.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the ``.mlt`` project.
    reference : str
        Regular expression matching the file names of the reference camera
        (``"L"``).
    others : dict[str, str]
        Label -> regular expression of the file names of other sources (``"W"``:
        WhatsApp videos); any other file is camera ``"A"``.

    Attributes
    ----------
    fps : float
        Frame rate.
    clock : MltClock
        Clock of the project.
    segments : dict[str, list[tuple[int, int, str, int]]]
        For each camera, ``(start, end, chain_id, t0)`` of each video clip on
        the timeline, in frames (`end` excluded); `t0` is the timeline position
        of the first frame of the source file (possibly negative).
    end : int
        Length of the project, in frames.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        reference: str = LUMIX,
        others: dict[str, str] = OTHERS,
    ):
        self.path = Path(path)
        self.dir = self.path.parent
        self.reference = re.compile(reference, re.I)
        self.others = {k: re.compile(v, re.I) for k, v in others.items()}
        self.tree = ET.parse(self.path)
        root = self.tree.getroot()
        self.clock = _clock_of(root)
        self.fps = self.clock.fps
        self.chains = {c.get("id"): c for c in root if c.tag in ("chain", "producer")}
        self.segments = {"L": [], "A": [], **{k: [] for k in self.others}}
        self.end = 0
        for pl in root.iter("playlist"):
            if _prop(pl, "shotcut:name") is None:
                continue
            pos = 0
            for el in pl:
                if el.tag == "blank":
                    pos += self.clock.frames(el.get("length"))
                elif el.tag == "entry":
                    i = self.clock.frames(el.get("in"))
                    n = self.clock.frames(el.get("out")) - i + 1
                    if _prop(pl, "shotcut:video") == "1":
                        cam = self.camera(self.resource(el.get("producer")))
                        self.segments[cam].append(
                            (pos, pos + n, el.get("producer"), pos - i)
                        )
                    pos += n
            self.end = max(self.end, pos)

    def camera(self, resource: str) -> str:
        """Label of the source of a file: ``"L"`` (reference), a key of `others`, else ``"A"``."""
        if self.reference.search(resource or ""):
            return "L"
        return next(
            (k for k, r in self.others.items() if r.search(resource or "")), "A"
        )

    def resource(self, chain_id: str) -> str:
        """File (``resource``) of a chain."""
        return _prop(self.chains[chain_id], "resource")

    def rushes(self) -> list[tuple[str, int]]:
        """Source files of the video tracks with their `t0`, in timeline order."""
        out = {}
        for segs in self.segments.values():
            for _, _, cid, t0 in segs:
                out.setdefault(self.resource(cid), t0)
        return sorted(out.items(), key=lambda kv: kv[1])

    def covering(self, cam: str, a: int, b: int) -> tuple[str, int] | None:
        """``(chain_id, t0)`` of the clip of `cam` covering frames ``[a, b)``, if any."""
        for s, e, cid, t0 in self.segments[cam]:
            if s <= a and b <= e:
                return cid, t0
        return None

    @property
    def video_end(self) -> int:
        """Last frame filmed by a camera, plus one (an audio track may be longer)."""
        return max(e for segs in self.segments.values() for _, e, _, _ in segs)

    def borders(self) -> list[int]:
        """Frames where a video clip starts or ends."""
        return sorted(
            {
                x
                for segs in self.segments.values()
                for s, e, _, _ in segs
                for x in (s, e)
            }
        )

    def crossfades(self) -> list[tuple[float, float, set[str]]]:
        """Cross-fades of the audio tracks: ``(start_s, end_s, files)``."""
        root = self.tree.getroot()
        tractors = {t.get("id"): t for t in root.iter("tractor")}
        out = []
        for pl in root.iter("playlist"):
            if _prop(pl, "shotcut:audio") != "1":
                continue
            pos = 0
            for el in pl:
                if el.tag == "blank":
                    pos += self.clock.frames(el.get("length"))
                elif el.tag == "entry":
                    n = (
                        self.clock.frames(el.get("out"))
                        - self.clock.frames(el.get("in"))
                        + 1
                    )
                    if el.get("producer") in tractors:
                        files = {
                            self.resource(tr.get("producer"))
                            for tr in tractors[el.get("producer")].findall("track")
                        }
                        out.append((pos / self.fps, (pos + n) / self.fps, files))
                    pos += n
        return out


# --------------------------------------------------------------------------------------------------
# Synchronization


def resync_project(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    reference: str = LUMIX,
) -> list[str]:
    """Write a copy of a project where every source is placed from the soundtracks.

    The reference file (the only one matching `reference`) stays where its picture
    is. Every other file is placed, to the nearest frame, where
    :func:`~mediautils.audiosync.sync_videos` says it belongs relative to it. All
    edit points (cuts and cross-fades of the audio tracks, filters) stay at their
    place on the timeline: only which frame of each source falls there changes. If
    a source has no frame for a border, the border moves and the neighbouring item
    is lengthened accordingly.

    Parameters
    ----------
    src : str | os.PathLike
        The synchronization project (not modified).
    dst : str | os.PathLike
        The new project (must not exist).
    reference : str
        Regular expression matching the reference file name.

    Returns
    -------
    list[str]
        A report: one line per source (offset, confidence, old and new position).
    """
    from mediautils.audiosync import sync_videos

    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    C = _clock_of(root)
    chains = {c.get("id"): c for c in root if c.tag in ("chain", "producer")}
    resource = {k: _prop(c, "resource") for k, c in chains.items()}
    last = {k: C.frames(c.get("out")) for k, c in chains.items() if c.get("out")}
    tractors = {
        t.get("id"): t
        for t in root.iter("tractor")
        if t.find("track") is not None
        and all(tr.get("producer") in chains for tr in t.findall("track"))
    }
    playlists = []
    for pl in root.iter("playlist"):
        if _prop(pl, "shotcut:name") is None:
            continue
        items, pos = [], 0
        for el in pl:
            if el.tag == "blank":
                n = C.frames(el.get("length"))
                items.append({"el": el, "kind": "blank", "S": pos, "E": pos + n})
                pos += n
            elif el.tag == "entry":
                i, o = C.frames(el.get("in")), C.frames(el.get("out"))
                p = el.get("producer")
                if p in tractors:
                    tracks = [
                        (tr, tr.get("producer"), pos - C.frames(tr.get("in")))
                        for tr in tractors[p].findall("track")
                    ]
                    kind = "tractor"
                else:
                    tracks, kind = [(el, p, pos - i)], "entry"
                items.append(
                    {
                        "el": el,
                        "kind": kind,
                        "S": pos,
                        "E": pos + o - i + 1,
                        "tracks": tracks,
                    }
                )
                pos += o - i + 1
        playlists.append((pl, items))

    ref_re = re.compile(reference, re.I)
    t0_now = {}
    ref_video = set()
    for _, items in playlists:
        for it in items:
            for _, p, t0 in it.get("tracks", []):
                t0_now.setdefault(resource[p], set()).add(t0)
                if (
                    ref_re.search(resource[p] or "")
                    and _prop(chains[p], "video_index") == "0"
                ):
                    ref_video.add(t0)
    refs = [r for r in t0_now if ref_re.search(r or "")]
    if len(refs) != 1 or len(ref_video) != 1:
        raise ValueError(
            f"expected one reference file placed once on a video track: {refs}, {ref_video}"
        )
    ref, ref_t0 = refs[0], ref_video.pop()
    target = {ref: ref_t0}
    report = [f"{ref}: reference, t0 = {ref_t0} (now {sorted(t0_now[ref])})"]
    for r in t0_now:
        if r == ref:
            continue
        s = sync_videos(src.parent / ref, src.parent / r)
        target[r] = ref_t0 + round(s.offset * C.fps)
        report.append(
            f"{r}: offset {s.offset:+.3f} s (confidence {s.confidence:.1f}, checks "
            f"{' '.join(f'{c * 1000:+.0f}' for c in s.checks)} ms), t0 {sorted(t0_now[r])} -> "
            f"{target[r]} (ideal {ref_t0 + s.offset * C.fps:.2f})"
        )

    for pl, items in playlists:
        for _ in range(100):
            changed = False
            if items and items[0]["kind"] != "blank":
                # the track starts on a source placed later: a blank at its head
                lack = max(
                    target[resource[p]] - items[0]["S"]
                    for _, p, _ in items[0]["tracks"]
                )
                if lack > 0:
                    el = ET.Element("blank", length=C.string(0))
                    first = next(
                        i for i, x in enumerate(pl) if x.tag in ("blank", "entry")
                    )
                    pl.insert(first, el)
                    items.insert(
                        0,
                        {
                            "el": el,
                            "kind": "blank",
                            "S": items[0]["S"],
                            "E": items[0]["S"],
                        },
                    )
            for k, it in enumerate(items):
                for _, p, _ in it.get("tracks", []):
                    t0 = target[resource[p]]
                    lack = t0 - it["S"]
                    if lack > 0:
                        if k == 0:
                            raise ValueError(
                                "a track starts on a source without frames there"
                            )
                        it["S"] += lack
                        items[k - 1]["E"] += lack
                        changed = True
                    excess = it["E"] - 1 - t0 - last[p]
                    if excess > 0:
                        it["E"] -= excess
                        if k + 1 < len(items):
                            items[k + 1]["S"] -= excess
                        changed = True
            if not changed:
                break
        else:  # pragma: no cover
            raise RuntimeError("could not place the borders")

    uses = {}
    for _, items in playlists:
        for it in items:
            n = it["E"] - it["S"]
            if it["kind"] == "blank":
                it["el"].set("length", C.string(n))
                continue
            if it["kind"] == "tractor":
                it["el"].set("in", C.string(0))
                it["el"].set("out", C.string(n - 1))
                t = tractors[it["el"].get("producer")]
                t.set("in", C.string(0))
                t.set("out", C.string(n - 1))
                for tr in t.findall("transition"):
                    tr.set("out", C.string(n - 1))
            for el, p, _ in it["tracks"]:
                i = it["S"] - target[resource[p]]
                el.set("in", C.string(i))
                el.set("out", C.string(i + n - 1))
                lo, hi = uses.get(p, (i, i + n - 1))
                uses[p] = (min(lo, i), max(hi, i + n - 1))
    for p, (lo, hi) in uses.items():
        for f in chains[p].findall("filter"):
            if f.get("in") is not None or lo > 0:
                f.set("in", C.string(lo))
            f.set("out", C.string(hi))
    _write(tree, dst)
    return report


# --------------------------------------------------------------------------------------------------
# Looking at the song


def extract_frames(
    project: SyncProject, out_dir: str | os.PathLike, width: int = 640
) -> None:
    """Extract one image per whole second of timeline for each camera (requires ffmpeg).

    Images are written as ``out_dir/<cam>/<cam>_<seconds>.jpg``.
    """
    for cam, segs in project.segments.items():
        d = Path(out_dir) / cam
        d.mkdir(parents=True, exist_ok=True)
        for s, e, cid, t0 in segs:
            first = int(np.ceil(s / project.fps))
            last = int((e - 1) / project.fps)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{first - t0 / project.fps:.3f}",
                    "-i",
                    str(project.dir / project.resource(cid)),
                    "-t",
                    f"{last - first + 1:.3f}",
                    "-vf",
                    f"fps=1,scale={width}:-2",
                    "-q:v",
                    "3",
                    "-start_number",
                    str(first),
                    str(d / f"{cam}_%04d.jpg"),
                ],
                check=True,
            )


def contact_sheets(
    frames_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    duration: float,
    step: int = 2,
    rows: int = 12,
    cols: int = 2,
) -> list[Path]:
    """Contact sheets of the images of :func:`extract_frames`: camera A | camera L per instant.

    Returns
    -------
    list[pathlib.Path]
        The sheets written in `out_dir`.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:  # pragma: no cover
        font = ImageFont.load_default()
    W, H = 400, 225
    times = list(range(0, int(duration) + 1, step))
    per = cols * rows
    out = []
    for k0 in range(0, len(times), per):
        chunk = times[k0 : k0 + per]
        sheet = Image.new("RGB", (cols * (2 * W + 30), rows * (H + 6)), "white")
        d = ImageDraw.Draw(sheet)
        for k, t in enumerate(chunk):
            c, r = divmod(k, rows)
            x, y = c * (2 * W + 30), r * (H + 6)
            for j, cam in enumerate("AL"):
                p = Path(frames_dir) / cam / f"{cam}_{t:04d}.jpg"
                im = (
                    Image.open(p).resize((W, H))
                    if p.exists()
                    else Image.new("RGB", (W, H), (40, 40, 40))
                )
                sheet.paste(im, (x + j * (W + 4), y))
            d.rectangle((x + 2, y + 2, x + 92, y + 30), fill="black")
            d.text((x + 6, y + 4), f"{t // 60}:{t % 60:02d}", fill="yellow", font=font)
        p = Path(out_dir) / f"sheet_{chunk[0]:03d}.jpg"
        sheet.save(p, quality=85)
        out.append(p)
    return out


def timeline_beats(project: SyncProject, sample_rate: int = 22050) -> np.ndarray:
    """Beats of the song (seconds of timeline), from the soundtrack of the cameras.

    Requires ffmpeg and librosa (extra ``concert``). The soundtrack is the
    reference camera where it films, else the other one.
    """
    import librosa

    y = np.zeros(int(project.end / project.fps * sample_rate) + sample_rate, np.float32)
    for cam in ("A", "L"):  # the reference camera last: it wins
        for s, e, cid, t0 in project.segments[cam]:
            raw = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{(s - t0) / project.fps:.3f}",
                    "-t",
                    f"{(e - s) / project.fps:.3f}",
                    "-i",
                    str(project.dir / project.resource(cid)),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-f",
                    "f32le",
                    "-",
                ],
                capture_output=True,
                check=True,
            ).stdout
            x = np.frombuffer(raw, np.float32)
            a = int(s / project.fps * sample_rate)
            y[a : a + len(x)] = x[: len(y) - a]
    _, b = librosa.beat.beat_track(y=y, sr=sample_rate, tightness=100)
    return librosa.frames_to_time(b, sr=sample_rate)


# --------------------------------------------------------------------------------------------------
# Montage


def nervous_shots(
    sections: list[tuple[float, float, list[str]]],
    lengths: tuple[float, ...] = (8, 4, 6, 4, 8, 6, 4),
) -> list[tuple[float, str]]:
    """Shots of 4 to 8 seconds, cycling through the shots chosen for each section.

    Parameters
    ----------
    sections : list[tuple[float, float, list[str]]]
        ``(start_s, end_s, cycle)``: in this part of the song, the shots follow
        `cycle` (names of cameras ``"A"``, ``"L"`` or of zooms).
    lengths : tuple[float, ...]
        Successive shot lengths, in seconds, cycled over the whole song.

    Returns
    -------
    list[tuple[float, str]]
        ``(start_s, shot)``, to give to :func:`build_montage`.

    Examples
    --------
        >>> nervous_shots([(0, 10, ["A"]), (10, 22, ["L", "zoom"])])
        [(0, 'A'), (10, 'L'), (14, 'zoom')]

    A section ends with a shot of at least 2 seconds: the last shot of the
    first section above lasts 10 s rather than leaving 2 s to a second one.
    """
    shots, k = [], 0
    for a, b, cycle in sections:
        t, i = a, 0
        while t < b - 2:
            shots.append((t, cycle[i % len(cycle)]))
            t += lengths[k % len(lengths)]
            i += 1
            k += 1
    return shots


def _affine(fid, cx, cy, z):
    """Shotcut "Size, Position & Rotate" filter: fixed zoom `z` centred on (cx, cy) of a 1920x1080 frame."""
    w, h = 1920 / z, 1080 / z
    x0 = min(max(cx - w / 2, 0), 1920 - w)
    y0 = min(max(cy - h / 2, 0), 1080 - h)
    filt = ET.Element("filter", id=fid)
    for k, v in [
        ("background", "color:#00000000"),
        ("mlt_service", "affine"),
        ("shotcut:filter", "affineSizePosition"),
        ("transition.fill", "1"),
        ("transition.distort", "0"),
        (
            "transition.rect",
            f"{-x0 * z:.1f} {-y0 * z:.1f} {1920 * z:.1f} {1080 * z:.1f} 1",
        ),
        ("transition.valign", "middle"),
        ("transition.halign", "center"),
        ("shotcut:animIn", "00:00:00.000"),
        ("shotcut:animOut", "00:00:00.000"),
        ("transition.threads", "0"),
    ]:
        ET.SubElement(filt, "property", name=k).text = v
    return filt


def build_montage(
    project: SyncProject,
    shots: list[tuple[float, str]],
    zooms: dict[str, tuple[str, float, float, float]],
    beats: np.ndarray | list[float],
    dst: str | os.PathLike,
    cut_end: float | None = None,
    min_shot: float = 2.5,
    overwrite: bool = False,
    trim_audio: bool = True,
    blocked: dict[str, list[tuple[float, float | None]]] | None = None,
) -> list[list]:
    """Write a montage of a synchronization project: cuts between cameras and fixed zooms.

    Only the video tracks change. The reference camera (``"L"``) stays all along on
    its track, split where it is zoomed; the other camera (``"A"``, on the track
    above) is shown where the montage chooses it, blanks elsewhere. Cuts are moved
    to the nearest beat. A shot on a camera that does not film at that moment (or
    is `blocked`) falls back on the other camera, and frames that no camera may show
    are left black (camera ``None`` in the result); shots shorter than `min_shot`
    are absorbed by a neighbour.

    Parameters
    ----------
    project : SyncProject
        The synchronization project.
    shots : list[tuple[float, str]]
        ``(start_s, shot)`` where `shot` is ``"A"``, ``"L"`` or a key of `zooms`
        (e.g. from :func:`nervous_shots`).
    zooms : dict[str, tuple[str, float, float, float]]
        Name -> ``(camera, centre_x, centre_y, factor)``, in pixels of the
        1920x1080 source.
    beats : array-like
        Beat times (seconds), e.g. from :func:`timeline_beats`.
    dst : str | os.PathLike
        The new project.
    cut_end : float | None
        If given, every track is cut there (seconds). By default, the project is
        cut at the last frame filmed by a camera (an audio track may be longer).
    min_shot : float
        Minimum length of a shot, in seconds.
    overwrite : bool
        Allow overwriting `dst`.
    trim_audio : bool
        With `cut_end`, also cut the audio tracks (set to ``False`` when they are
        about to be replaced by :func:`use_single_audio`).
    blocked : dict[str, list[tuple[float, float | None]]] | None
        Camera -> ``[(start_s, end_s)]`` where it must not be shown (someone walks
        in front of it...); ``end_s = None``: until the end.

    Returns
    -------
    list[list]
        The shots actually made: ``[start, end, camera, zoom]`` in frames.
    """
    dst = Path(dst)
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    P, C = project, project.clock
    end = min(P.end, P.video_end) if cut_end is None else round(cut_end * P.fps)
    bt = np.asarray(beats, dtype=float)

    def snap(t):
        if len(bt) and bt[0] - 1 < t < bt[-1] + 1:
            return float(bt[np.argmin(np.abs(bt - t))])
        return t

    hidden = {
        cam: [
            (round(a * P.fps), end if b is None else round(b * P.fps)) for a, b in ivs
        ]
        for cam, ivs in (blocked or {}).items()
    }

    def films(cam, a, b):
        """Whether `cam` films frames [a, b) and may be shown there."""
        return (
            cam is not None
            and P.covering(cam, a, b) is not None
            and not any(a < hb and ha < b for ha, hb in hidden.get(cam, []))
        )

    borders = set(P.borders()) | {
        x for ivs in hidden.values() for iv in ivs for x in iv
    }
    cuts = [(round(snap(t) * P.fps) if t > 0 else 0, s) for t, s in shots]
    segs = []
    for k, (a, s) in enumerate(cuts):
        b = min(cuts[k + 1][0] if k + 1 < len(cuts) else end, end)
        if b <= a:
            continue
        cam, zoom = (zooms[s][0], s) if s in zooms else (s, None)
        pts = sorted({a, b} | {x for x in borders if a < x < b})
        for x, y in zip(pts, pts[1:]):
            if films(cam, x, y):
                segs.append([x, y, cam, zoom])
            else:  # the first other source that may be shown, else black (camera None)
                other = next(
                    (c for c in P.segments if c != cam and films(c, x, y)), None
                )
                segs.append([x, y, other, None])
    # First glue back the pieces of a same shot (split at the borders of files or of `blocked`
    # intervals), so that a piece is not mistaken for a short shot and absorbed.
    glued = []
    for seg in segs:
        if glued and glued[-1][2:] == seg[2:] and glued[-1][1] == seg[0]:
            glued[-1][1] = seg[1]
        else:
            glued.append(seg)
    merged = []
    for seg in glued:
        if merged and merged[-1][2:] == seg[2:] and merged[-1][1] == seg[0]:
            merged[-1][1] = seg[1]
        elif (
            merged
            and seg[1] - seg[0] < min_shot * P.fps
            and films(merged[-1][2], merged[-1][0], seg[1])
        ):
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    k = 0
    ms = round(min_shot * P.fps)
    while (
        k < len(merged) - 1
    ):  # a short shot the previous one could not absorb goes to the next one,
        a, b = merged[k][:2]  # or else lengthens at the expense of the next one
        nxt = merged[k + 1]
        if b - a < ms and films(nxt[2], a, nxt[1]):
            nxt[0] = a
            del merged[k]
            continue
        if b - a < ms and films(merged[k][2], a, a + ms) and nxt[1] - (a + ms) >= ms:
            merged[k][1] = nxt[0] = a + ms
        k += 1

    tree = copy.deepcopy(P.tree)
    root = tree.getroot()
    chains = {c.get("id"): c for c in root if c.tag in ("chain", "producer")}
    n_zoom = 0

    def chain_for(cid, zoom, a, b):
        nonlocal n_zoom
        if zoom is None:
            return cid
        n_zoom += 1
        c = copy.deepcopy(chains[cid])
        c.set("id", f"montage_zoom{n_zoom}")
        _, cx, cy, z = zooms[zoom]
        filt = _affine(f"montage_affine{n_zoom}", cx, cy, z)
        filt.set("in", C.string(a))
        filt.set("out", C.string(b))
        c.append(filt)
        root.insert(list(root).index(chains[cid]) + 1, c)
        return c.get("id")

    def fill(pl, items):
        for el in list(pl):
            if el.tag in ("entry", "blank"):
                pl.remove(el)
        pos = 0
        for a, b, cid, t0, zoom in items:
            if a > pos:
                ET.SubElement(pl, "blank", length=C.string(a - pos))
            sa, sb = a - t0, b - 1 - t0
            ET.SubElement(
                pl,
                "entry",
                producer=chain_for(cid, zoom, sa, sb),
                **{"in": C.string(sa), "out": C.string(sb)},
            )
            pos = b

    # Frozen list: chain_for inserts zoom chains in the root, which would derail a live iterator.
    for pl in [pl for pl in root.iter("playlist") if _prop(pl, "shotcut:video") == "1"]:
        entries = pl.findall("entry")
        if not entries:
            continue
        cam = P.camera(P.resource(entries[0].get("producer")))
        items = []
        if cam == "L":
            # The reference camera lies under camera A: plain wherever L or A is shown (so that the
            # cuts of A need no cut here), zoomed where chosen zoomed, and absent elsewhere: where no
            # camera may be shown (black), and under other sources (e.g. a vertical phone video,
            # shown with black bars rather than over the reference).
            for s, e, cid, t0 in P.segments["L"]:
                for a, b, c, zoom in merged:
                    a, b = max(a, s), min(b, e, end)
                    if b <= a or c not in ("L", "A"):
                        continue
                    z = zoom if c == "L" else None
                    if (
                        items
                        and items[-1][4] is None
                        and z is None
                        and items[-1][1] == a
                    ):
                        items[-1] = (items[-1][0], b, cid, t0, None)
                    else:
                        items.append((a, b, cid, t0, z))
        else:
            for a, b, c, zoom in merged:
                if c == cam:
                    cid, t0 = P.covering(cam, a, b)
                    items.append((a, b, cid, t0, zoom))
        fill(pl, items)
    if end < P.end:
        _trim_end(root, C, end, audio=trim_audio)
    _write(tree, dst)
    return merged


def _trim_end(root, C, end, audio=True):
    tractors = {t.get("id") for t in root.iter("tractor")}
    for pl in root.iter("playlist"):
        if _prop(pl, "shotcut:name") is None and pl.get("id") != "background":
            continue
        if not audio and _prop(pl, "shotcut:audio") == "1":
            continue
        pos, cut = 0, False
        for el in list(pl):
            if el.tag not in ("entry", "blank"):
                continue
            if cut:
                pl.remove(el)
                continue
            n = (
                C.frames(el.get("length"))
                if el.tag == "blank"
                else C.frames(el.get("out")) - C.frames(el.get("in")) + 1
            )
            if pos + n > end:
                if el.tag == "entry" and el.get("producer") in tractors:
                    raise ValueError("cut inside a cross-fade")
                if el.tag == "blank":
                    el.set("length", C.string(end - pos))
                else:
                    el.set("out", C.string(C.frames(el.get("in")) + end - pos - 1))
                cut = True
            pos += n
            if pos >= end:
                cut = True
    for p in root:
        if p.get("id") == "black":
            p.set("out", C.string(end - 1))
            p.find("property[@name='length']").text = C.string(end)
    for t in root.iter("tractor"):
        if _prop(t, "shotcut") == "1":
            t.set("out", C.string(end - 1))
    for pl in root.iter("playlist"):
        for f in pl.findall("filter"):
            if f.get("out") and C.frames(f.get("out")) > end - 1:
                f.set("out", C.string(end - 1))


def trim_project_end(path: str | os.PathLike, end_s: float, audio: bool = True) -> None:
    """Cut every track of a project at `end_s` seconds (in place).

    The cut must not fall inside a cross-fade of a track that is cut.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    C = _clock_of(root)
    _trim_end(root, C, round(end_s * C.fps), audio=audio)
    _write(tree, path)


def trim_project_head(path: str | os.PathLike, start_s: float) -> None:
    """Remove the first `start_s` seconds of every track of a project (in place).

    Meant for generated projects whose audio is a single file
    (:func:`use_single_audio`): there must be no cross-fade in the removed part.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    C = _clock_of(root)
    S = round(start_s * C.fps)
    tractors = {t.get("id") for t in root.iter("tractor")}
    for pl in root.iter("playlist"):
        if _prop(pl, "shotcut:name") is None and pl.get("id") != "background":
            continue
        left = S
        for el in list(pl):
            if left <= 0 or el.tag not in ("entry", "blank"):
                continue
            if el.tag == "blank":
                n = C.frames(el.get("length"))
                if n <= left:
                    pl.remove(el)
                else:
                    el.set("length", C.string(n - left))
            else:
                if el.get("producer") in tractors:
                    raise ValueError("cross-fade in the removed head")
                i, o = C.frames(el.get("in")), C.frames(el.get("out"))
                n = o - i + 1
                if n <= left:
                    pl.remove(el)
                else:
                    el.set("in", C.string(i + left))
            left -= n
    for p in root:
        if p.get("id") == "black":
            p.set("out", C.string(C.frames(p.get("out")) - S))
            length = p.find("property[@name='length']")
            length.text = C.string(C.frames(length.text) - S)
    for t in root.iter("tractor"):
        if _prop(t, "shotcut") == "1" and t.get("out"):
            t.set("out", C.string(C.frames(t.get("out")) - S))
    for pl in root.iter("playlist"):
        for f in pl.findall("filter"):
            if f.get("out"):
                f.set("out", C.string(max(0, C.frames(f.get("out")) - S)))
    _write(tree, path)


def _duration(path):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    return float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )


def use_single_audio(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    sound: str,
    track_name: str = "Son mix Claude",
) -> None:
    """Write a copy of a project whose sound is a single stereo file, from the start.

    The first audio track gets `sound`; the other audio tracks are emptied (renamed
    ``(vide)``); their cross-fades, pan filters and the chains no longer used are
    removed. Video tracks are unchanged.

    Parameters
    ----------
    src : str | os.PathLike
        The project (not modified).
    dst : str | os.PathLike
        The new project (must not exist); `sound` must be next to it.
    sound : str
        File name of the sound, relative to the folder of `dst`.
    track_name : str
        Name of the audio track holding it.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    C = _clock_of(root)
    audio = [pl for pl in root.iter("playlist") if _prop(pl, "shotcut:audio") == "1"]
    removed = set()
    for pl in audio:
        for el in list(pl):
            if el.tag in ("entry", "blank", "filter"):
                if el.tag == "entry":
                    removed.add(el.get("producer"))
                pl.remove(el)
    for t in [t for t in root if t.tag == "tractor" and t.get("id") in removed]:
        removed |= {tr.get("producer") for tr in t.findall("track")}
        root.remove(t)
    text = ET.tostring(root, encoding="unicode")
    template = None
    for c in [c for c in root if c.tag == "chain" and c.get("id") in removed]:
        template = template or copy.deepcopy(c)
        if f'producer="{c.get("id")}"' not in text:
            root.remove(c)
    if template is None:
        raise ValueError("no audio clip to replace")
    n = int(round(_duration(dst.parent / sound) * C.fps))
    c = template
    c.set("id", "single_audio")
    c.set("out", C.string(n - 1))
    for f in c.findall("filter"):
        c.remove(f)
    props = {p.get("name"): p for p in c.findall("property")}
    props["resource"].text = sound
    props["audio_index"].text = "0"
    props["video_index"].text = "-1"
    props["length"].text = C.string(n)
    if "shotcut:caption" in props:
        props["shotcut:caption"].text = sound
    if "shotcut:hash" in props:
        c.remove(props["shotcut:hash"])
    root.insert(list(root).index(audio[0]), c)
    # Never longer than the project (the black background): past the last picture, Shotcut and melt show
    # a white frame.
    black = next((p for p in root if p.get("id") == "black"), None)
    if black is not None:
        n_used = min(n, C.frames(black.find("property[@name='length']").text))
    else:  # pragma: no cover
        n_used = n
    ET.SubElement(
        audio[0],
        "entry",
        producer="single_audio",
        **{"in": C.string(0), "out": C.string(n_used - 1)},
    )
    audio[0].find("property[@name='shotcut:name']").text = track_name
    for pl in audio[1:]:
        pl.find("property[@name='shotcut:name']").text = "(vide)"
    _write(tree, dst)


def export_project(
    project: str | os.PathLike,
    dst: str | os.PathLike,
    melt: str | os.PathLike = r"C:\Program Files\Shotcut\melt.exe",
    crf: int = 23,
) -> None:
    """Render a Shotcut project to an MP4 with melt (Shotcut's engine), as Shotcut would.

    Default quality = Shotcut's default (quality 55 %, i.e. x264 crf 23), AAC 384 kb/s.
    """
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    subprocess.run(
        [
            str(melt),
            "-quiet",
            str(project),
            "-consumer",
            f"avformat:{dst}",
            "f=mp4",
            "vcodec=libx264",
            f"crf={crf}",
            "preset=medium",
            "g=300",
            "bf=3",
            "pix_fmt=yuv420p",
            "acodec=aac",
            "ab=384k",
            "ar=48000",
            "channels=2",
            "movflags=+faststart",
            "real_time=-8",
        ],
        check=True,
        capture_output=True,
    )
