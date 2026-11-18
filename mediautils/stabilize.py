"""Stabilization of a video: on a stand that shakes, or held by hand.

:func:`stabilize_video` — a stand that shakes (someone taps the beat on it): the shift of
each picture is measured on still background (e.g. the top band of the picture, above the
heads: ceiling, curtains), by phase correlation against the median picture, on two zones
that must agree (else somebody walks into the band: the shift is interpolated). Each
picture is then moved back; a thin black border appears where the picture is missing
(hidden by any zoom of 1.03 or more). A special case, not a step of every montage.

:func:`stabilize_handheld` — a phone held by hand on a scene that stays in view (children
boxing on a mat, 2026-11-16): the camera drifts slowly (70 px in 11 s) and turns a little
(±0.35°). Choices, tried on that video with the user:

- a **similarity** (shift, rotation, zoom) of each picture onto the first one: the picture
  stays flat. A homography fitted on the floor nailed the mat to the pixel but bent the
  top of the picture from one frame to the next (« une feuille de papier qu'un ventilateur
  fait bouger »);
- measured on points of the **whole picture** (floor and background): fitted on the floor
  only, the noise of the rotation is magnified far from it (top 3x shakier than the
  original). Floor and background are not at the same distance: a flat picture cannot nail
  both, the remaining slow drift (~15 px) is shared;
- each picture registered **directly onto the first one** (starting from the motion
  chained frame to frame), so that nothing accumulates; then the trajectory is **smoothed**
  over a few frames (σ = 3): the measurement noise goes, the drift of the camera does not
  come back. Residual shake 0.3 px per frame (the original 0.5), drift 70 px → ~15 px;
- one single resampling from the original (stabilization, crop and zoom together, Lanczos,
  plane by plane in YUV: no colour conversion), sound copied. A rotation costs nothing
  more than a shift: every pixel is resampled anyway.
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def estimate_shifts(frames, zones, reference=None) -> np.ndarray:
    """Shift ``(dx, dy)`` of each frame against the reference, measured on each zone.

    Parameters
    ----------
    frames : array-like
        Grey pictures (same size), e.g. a band of each frame of a video.
    zones : sequence of (int, int)
        Column ranges where the picture is still background.
    reference : numpy.ndarray, optional
        Reference picture; default the median of the frames.

    Returns
    -------
    numpy.ndarray
        Shape ``(zones, frames, 2)``.
    """
    import cv2

    frames = np.asarray(frames)
    ref = (
        np.median(frames[:: max(len(frames) // 60, 1)], axis=0)
        if reference is None
        else reference
    ).astype(np.float32)
    h = frames.shape[1]
    out = []
    for a, b in zones:
        hann = cv2.createHanningWindow((b - a, h), cv2.CV_32F)
        r = ref[:, a:b]
        out.append(
            [
                cv2.phaseCorrelate(r, f[:, a:b].astype(np.float32), hann)[0]
                for f in frames
            ]
        )
    return np.array(out)


def combine_shifts(shifts, disagree: float = 1.5) -> tuple[np.ndarray, float]:
    """One shift per frame from the zones; frames where they disagree are interpolated.

    Returns the shifts (median removed) and the share of interpolated frames.

    >>> s = np.array([[[0, 1], [0, 2], [5, 9], [0, 4]], [[0, 1], [0, 2], [0, 3], [0, 4]]], float)
    >>> c, bad = combine_shifts(s)
    >>> (c[:, 1] + 2.5).tolist(), bad
    ([1.0, 2.0, 3.0, 4.0], 0.25)
    """
    s = np.asarray(shifts, float)
    shift = s.mean(axis=0)
    bad = (np.abs(s.max(axis=0) - s.min(axis=0)) > disagree).any(axis=1)
    good = ~bad
    for k in range(2):
        shift[bad, k] = np.interp(
            np.flatnonzero(bad), np.flatnonzero(good), shift[good, k]
        )
    return shift - np.median(shift, axis=0), float(bad.mean())


def stabilize_video(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    band: int = 110,
    zones=((0, 520), (720, 1480)),
    crf: int = 16,
    size: tuple[int, int] = (1920, 1080),
    fps: str = "60000/1001",
) -> dict:
    """Write a stabilized copy of a video (the sound copied unchanged; `src` untouched).

    Requires ffmpeg and OpenCV (extra ``concert``). About 25 min for 9 min of 1080p60.

    Parameters
    ----------
    src, dst : str | os.PathLike
        Source video, stabilized copy (must not exist).
    band : int
        Height of the top band of the picture where the shifts are measured.
    zones : sequence of (int, int)
        Columns of still background in that band (two zones, which must agree).
    crf : int
        x264 quality of the copy (16: visually lossless).
    size, fps : tuple, str
        Picture size and frame rate of the video.

    Returns
    -------
    dict
        Standard deviation and maximum of the shifts (pixels), share of interpolated
        frames, number of frames.
    """
    import cv2

    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    w, h = size
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            f"crop={w}:{band}:0:0,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    bands = []
    while len(buf := proc.stdout.read(w * band)) == w * band:
        bands.append(np.frombuffer(buf, np.uint8).reshape(band, w))
    shift, bad = combine_shifts(estimate_shifts(bands, zones))
    t0 = time.time()
    dec = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(src),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            fps,
            "-i",
            "-",
            "-i",
            str(src),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        stdin=subprocess.PIPE,
    )
    n = 0
    while len(buf := dec.stdout.read(w * h * 3)) == w * h * 3:
        dx, dy = shift[min(n, len(shift) - 1)]
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        enc.stdin.write(
            cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR).tobytes()
        )
        n += 1
    enc.stdin.close()
    enc.wait()
    return {
        "std_px": shift.std(axis=0).round(2).tolist(),
        "max_px": np.abs(shift).max(axis=0).round(1).tolist(),
        "interpolated": bad,
        "frames": n,
        "seconds": round(time.time() - t0),
    }


# --- Held by hand -------------------------------------------------------------------------


def _probe(src) -> dict:
    """Size of the decoded pictures (rotation applied), pixel format, frame rate, colours, sound."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate,color_range,color_space,"
            "color_transfer,color_primaries:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    rot = int(s.get("tags", {}).get("rotate", 0))
    for sd in s.get("side_data_list", []):
        rot = int(sd.get("rotation", rot))
    w, h = s["width"], s["height"]
    if rot % 180:
        w, h = h, w
    audio = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    colour = {
        k: s.get(k)
        for k in ("color_range", "color_space", "color_transfer", "color_primaries")
    }
    return {
        "size": (w, h),
        "pix_fmt": s.get("pix_fmt"),
        "fps": s["r_frame_rate"],
        "colour": {k: v for k, v in colour.items() if v and v != "unknown"},
        "audio": audio,
    }


def _raw_frames(src, size, pix_fmt):
    """Decoded pictures of a video, as flat ``uint8`` arrays (``gray`` or 4:2:0)."""
    w, h = size
    n = w * h if pix_fmt == "gray" else w * h * 3 // 2
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(src),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        while len(buf := proc.stdout.read(n)) == n:
            yield np.frombuffer(buf, np.uint8)
    finally:
        proc.stdout.close()
        proc.wait()


def _gaussian_smooth(x, sigma):
    """Gaussian smoothing along the first axis, the edges repeated.

    >>> _gaussian_smooth(np.array([0.0, 0, 3, 0, 0]), 0).tolist()
    [0.0, 0.0, 3.0, 0.0, 0.0]
    >>> y = _gaussian_smooth(np.array([0.0, 0, 3, 0, 0]), 1)
    >>> bool(0 < y[2] < 3 and round(y[1], 9) == round(y[3], 9) > 0)
    True
    >>> _gaussian_smooth(np.ones((4, 2)), 2).tolist()
    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    """
    x = np.asarray(x, float)
    if sigma <= 0:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, axis=0), x, np.repeat(x[-1:], r, axis=0)])
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def _fit_similarity(a, b) -> np.ndarray:
    """Least-squares similarity (3x3) sending the points `a` onto the points `b`.

    >>> a = np.array([[0.0, 0], [10, 0], [0, 10]])
    >>> (_fit_similarity(a, a * 2 + [3, -1]).round(6) + 0.0).tolist()
    [[2.0, 0.0, 3.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append(([x, -y, 1, 0], u))
        rows.append(([y, x, 0, 1], v))
    m = np.array([r for r, _ in rows])
    p, *_ = np.linalg.lstsq(m, np.array([v for _, v in rows]), rcond=None)
    c, s, tx, ty = p
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def _smooth_motion(mats, size, sigma) -> np.ndarray:
    """Smooth a motion by smoothing the paths of three points of the picture.

    (Smoothing the shift apart from the rotation would be wrong: the shift depends on the
    centre of the rotation.)
    """
    if sigma <= 0:
        return np.array(mats, float)
    w, h = size
    ref = np.array([[0.15 * w, 0.15 * h], [0.85 * w, 0.15 * h], [0.5 * w, 0.88 * h]])
    paths = np.array([ref @ m[:2, :2].T + m[:2, 2] for m in mats])
    paths = _gaussian_smooth(paths, sigma)
    return np.array([_fit_similarity(ref, p) for p in paths])


def camera_motion(
    src: str | os.PathLike,
    smooth: float = 3.0,
    points: int = 2500,
    zone: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Motion of a camera held by hand: the similarity sending each picture onto the first one.

    Points of the whole first picture are followed directly into each picture (optical flow,
    starting from the motion chained picture to picture; checked back and forth), and a
    similarity (shift, rotation, zoom) is fitted on them (RANSAC: people moving are left
    out). Nothing accumulates from one picture to the next. Then the paths are smoothed over
    `smooth` frames (Gaussian σ), which removes the noise of the measurement. The scene must
    stay in view: a picture that cannot be registered on the first one takes the chained
    motion (a warning says how many).

    Requires ffmpeg and OpenCV (extra ``concert``).

    Parameters
    ----------
    src : str | os.PathLike
        The video.
    smooth : float
        σ of the smoothing, in frames (0: none).
    points : int
        Number of points followed.
    zone : (int, int, int, int), optional
        ``(x0, y0, x1, y1)`` in pixels of the first picture of `src` (as decoded, not
        mirrored): follow only the points of this part of the scene, e.g. the far background
        of a selfie (near things and the background do not move alike when the camera moves).
        It must stay in view.

    Returns
    -------
    numpy.ndarray
        Shape ``(frames, 3, 3)``: coordinates in picture n → coordinates in the first one.
    """
    import warnings

    import cv2

    info = _probe(src)
    w, h = info["size"]
    flow = {"winSize": (31, 31), "maxLevel": 3, "flags": cv2.OPTFLOW_USE_INITIAL_FLOW}
    g0 = prev = p0 = None
    acc = np.eye(3)
    mats, weak = [], 0
    for buf in _raw_frames(src, (w, h), "gray"):
        g = buf.reshape(h, w)
        if g0 is None:
            g0 = prev = g
            mask = None
            if zone is not None:
                mask = np.zeros_like(g0)
                mask[zone[1] : zone[3], zone[0] : zone[2]] = 255
            p0 = cv2.goodFeaturesToTrack(g0, points, 0.005, 12, mask=mask)
            if p0 is None or len(p0) < 50:
                raise ValueError(
                    f"too few details to follow in the first picture of {src}"
                )
            mats.append(np.eye(3))
            continue
        # picture to picture first (big moves): the starting point
        pp = cv2.goodFeaturesToTrack(prev, 800, 0.01, 15)
        if pp is not None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                prev, g, pp, None, winSize=(31, 31), maxLevel=4
            )
            ok = st.ravel() == 1
            if ok.sum() >= 10:
                m, _ = cv2.estimateAffinePartial2D(
                    nxt[ok], pp[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5
                )
                if m is not None:
                    acc = acc @ np.vstack([m, [0, 0, 1]])
        # then directly onto the first picture
        guess = cv2.transform(p0, np.linalg.inv(acc)[:2]).astype(np.float32)
        pn, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, p0, guess.copy(), **flow)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g, g0, pn, p0.copy(), **flow)
        ok = (
            (st.ravel() == 1)
            & (st2.ravel() == 1)
            & (np.linalg.norm((back - p0).reshape(-1, 2), axis=1) < 0.5)
        )
        m = None
        if ok.sum() >= 30:
            m, inl = cv2.estimateAffinePartial2D(
                pn[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5
            )
            if m is not None and inl.sum() >= 30:
                sel = inl.ravel() == 1
                m, _ = cv2.estimateAffinePartial2D(
                    pn[ok][sel], p0[ok][sel], method=cv2.LMEDS
                )
            else:
                m = None
        if m is None:
            weak += 1
        else:
            acc = np.vstack([m, [0, 0, 1]])
        mats.append(acc.copy())
        prev = g
    if weak:
        warnings.warn(
            f"{weak} pictures of {len(mats)} not registered on the first one (chained "
            "motion used): does the scene leave the view?",
            stacklevel=2,
        )
    return _smooth_motion(np.array(mats), (w, h), smooth)


def _mirror(mats, w, mirror) -> tuple[np.ndarray, np.ndarray]:
    """Motion of the mirrored video and the mirror to apply to each picture (identity if not).

    >>> mats, flip = _mirror(np.array([np.eye(3) + [[0, 0, 5], [0, 0, 0], [0, 0, 0]]]), 10, True)
    >>> mats[0][0].tolist(), flip[0].tolist()
    ([1.0, 0.0, -5.0], [-1.0, 0.0, 9.0])
    """
    if not mirror:
        return mats, np.eye(3)
    flip = np.array([[-1.0, 0, w - 1], [0, 1, 0], [0, 0, 1]])
    return flip @ mats @ flip, flip


def _follow(mats, size, follow, fps) -> np.ndarray:
    """Motion onto a camera following the slow moves (path smoothed over `follow` s), or `mats`.

    A slow drift of 10 px per picture, followed over a long time: only the gaps to its mean
    path are left.

    >>> mats = np.array([np.eye(3) + [[0, 0, x], [0, 0, 0], [0, 0, 0]] for x in (0, 10, 20)])
    >>> (_follow(mats, (1920, 1080), 100, "1/1")[:, 0, 2].round(1) + 0.0).tolist()
    [-9.9, 0.0, 9.9]
    >>> _follow(mats, (1920, 1080), 0, "30/1") is mats
    True
    """
    if not follow:
        return mats
    num, den = (int(v) for v in fps.split("/"))
    path = _smooth_motion(mats, size, follow * num / den)
    return np.linalg.inv(path) @ mats


def _frame_margin(mats, frame, size) -> tuple[float, int]:
    """Smallest distance (px) between the frame and the edges of the pictures, and where.

    >>> _frame_margin(np.array([np.eye(3)]), (10, 20, 1600), (1920, 1080))
    (10.0, 0)
    """
    w, h = size
    x0, y0, fw = frame
    fh = fw * h / w
    corners = np.array([[x0, y0], [x0 + fw, y0], [x0 + fw, y0 + fh], [x0, y0 + fh]])
    inv = np.linalg.inv(np.asarray(mats, float))
    q = np.einsum("nij,kj->nki", inv[:, :2, :2], corners) + inv[:, None, :2, 2]
    m = np.minimum.reduce([q[..., 0], q[..., 1], w - q[..., 0], h - q[..., 1]]).min(
        axis=1
    )
    return float(m.min()), int(m.argmin())


def _auto_frame(mats, size, margin=2.0) -> tuple[float, float, float]:
    """Largest frame (ratio of the picture) centred where the camera points on average, covered
    by every picture.

    >>> _auto_frame(np.array([np.eye(3)]), (1920, 1080), margin=0)
    (0.0, 0.0, 1920.0)
    """
    w, h = size
    mats = np.asarray(mats, float)
    cx, cy = (mats[:, :2, :2] @ np.array([w / 2, h / 2]) + mats[:, :2, 2]).mean(axis=0)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        frame = (cx - w * mid / 2, cy - h * mid / 2, w * mid)
        if _frame_margin(mats, frame, size)[0] >= margin:
            lo = mid
        else:
            hi = mid
    if lo < 0.3:
        raise ValueError(
            "the camera moves too much: no frame is covered by every picture"
        )
    return tuple(
        float(round(v, 1)) + 0.0 for v in (cx - w * lo / 2, cy - h * lo / 2, w * lo)
    )


def stabilize_handheld(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    frame: tuple[float, float, float] | None = None,
    smooth: float = 3.0,
    crf: int = 23,
    motion: np.ndarray | None = None,
    mirror: bool = False,
    follow: float = 0.0,
) -> dict:
    """Write a stabilized and cropped copy of a video held by hand (`src` untouched).

    The method and why are in the docstring of :mod:`mediautils.stabilize`. One single
    resampling from the original (motion, crop and zoom together, Lanczos, plane by plane
    in YUV: no colour conversion), to the size of the original; the sound and the metadata
    of the file (dates) copied; quality of Shotcut's default (crf 23).

    Requires ffmpeg and OpenCV (extra ``concert``). From a ``.bat``, run it at low priority
    (``start /low /affinity F000 /wait /b uv run ...``).

    Parameters
    ----------
    src, dst : str | os.PathLike
        Source video, copy (must not exist).
    frame : (float, float, float), optional
        What to show: ``(x0, y0, width)`` in pixels of the **first picture** of `src` (the
        height follows the ratio of the picture). Default: the largest frame centred where
        the camera points on average that every picture covers. A frame that some picture
        does not cover is refused (it would show black): move it or narrow it. Check it with
        :func:`stabilized_sheet`.
    smooth : float
        σ of the smoothing of the motion, in frames (see :func:`camera_motion`).
    crf : int
        x264 quality.
    motion : numpy.ndarray, optional
        ``camera_motion(src, smooth)``, measured once to try several frames.
    mirror : bool
        Also mirror the picture left-right (a selfie), in the same resampling; `frame` is
        then given in the mirrored first picture (`motion` stays the one of `src`).
    follow : float
        0: the camera is fixed on the scene. Otherwise, in seconds: the camera follows the
        slow moves of the video (its path smoothed over `follow` s, Gaussian σ) and only the
        shakes are removed; it crops much less when the video drifts (a selfie). `frame` is
        then given in pictures of this smoothed camera (close to the original pictures).

    Returns
    -------
    dict
        The frame, the zoom, the smallest margin left (px of the source, frame number),
        number of frames, seconds.
    """
    import cv2

    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    t0 = time.time()
    info = _probe(src)
    w, h = info["size"]
    mats = camera_motion(src, smooth) if motion is None else np.asarray(motion, float)
    mats, flip = _mirror(mats, w, mirror)
    mats = _follow(mats, (w, h), follow, info["fps"])
    if frame is None:
        frame = _auto_frame(mats, (w, h))
    margin, worst = _frame_margin(mats, frame, (w, h))
    if margin < 0:
        raise ValueError(
            f"the frame {frame} goes {-margin:.0f} px out of picture {worst}: move it or narrow it"
        )
    x0, y0, fw = frame
    z = w / fw
    crop = np.array([[z, 0, -x0 * z], [0, z, -y0 * z], [0, 0, 1.0]])
    half = np.diag([0.5, 0.5, 1.0])
    pix = info["pix_fmt"] if info["pix_fmt"] in ("yuv420p", "yuvj420p") else "yuv420p"
    tags = {
        "color_range": "-color_range",
        "color_space": "-colorspace",
        "color_transfer": "-color_trc",
        "color_primaries": "-color_primaries",
    }
    colour = [a for k, v in info["colour"].items() for a in (tags[k], v)]
    audio = []
    if info["audio"]:
        copy = info["audio"] in ("aac", "mp3", "ac3", "eac3", "opus", "alac")
        audio = [
            "-map",
            "1:a:0",
            "-c:a",
            *(["copy"] if copy else ["aac", "-b:a", "384k"]),
        ]
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-s",
            f"{w}x{h}",
            "-r",
            info["fps"],
            "-i",
            "-",
            "-i",
            str(src),
            "-map",
            "0:v",
            *audio,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            pix,
            *colour,
            "-map_metadata",
            "1",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        stdin=subprocess.PIPE,
    )
    lanczos = {"flags": cv2.INTER_LANCZOS4, "borderMode": cv2.BORDER_REPLICATE}
    n = 0
    for buf in _raw_frames(src, (w, h), pix):
        m = crop @ mats[min(n, len(mats) - 1)] @ flip
        mh = half @ m @ np.linalg.inv(half)
        y = buf[: w * h].reshape(h, w)
        u = buf[w * h : w * h * 5 // 4].reshape(h // 2, w // 2)
        v = buf[w * h * 5 // 4 :].reshape(h // 2, w // 2)
        enc.stdin.write(
            cv2.warpPerspective(y, m, (w, h), **lanczos).tobytes()
            + cv2.warpPerspective(u, mh, (w // 2, h // 2), **lanczos).tobytes()
            + cv2.warpPerspective(v, mh, (w // 2, h // 2), **lanczos).tobytes()
        )
        n += 1
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"ffmpeg failed writing {dst}")
    return {
        "frame": tuple(float(v) for v in frame),
        "zoom": round(z, 3),
        "margin_px": (round(margin, 1), worst),
        "frames": n,
        "seconds": round(time.time() - t0),
    }


def stabilized_sheet(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    frame: tuple[float, float, float] | None = None,
    smooth: float = 3.0,
    step: float = 0.5,
    cols: int = 4,
    width: int = 640,
    motion: np.ndarray | None = None,
    mirror: bool = False,
    follow: float = 0.0,
) -> Path:
    """Sheet of stabilized pictures (whole, in the coordinates of the first one) with the frame
    drawn in red, every `step` seconds: to check a frame before :func:`stabilize_handheld`.

    Parameters as in :func:`stabilize_handheld`; `width` of each picture on the sheet.
    """
    import cv2

    info = _probe(src)
    w, h = info["size"]
    mats = camera_motion(src, smooth) if motion is None else np.asarray(motion, float)
    mats, flip = _mirror(mats, w, mirror)
    mats = _follow(mats, (w, h), follow, info["fps"])
    if frame is None:
        frame = _auto_frame(mats, (w, h))
    x0, y0, fw = frame
    num, den = (int(v) for v in info["fps"].split("/"))
    every = max(round(step * num / den), 1)
    tiles = []
    for n, buf in enumerate(_raw_frames(src, (w, h), "yuv420p")):
        if n % every:
            continue
        bgr = cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
        img = cv2.warpPerspective(bgr, mats[min(n, len(mats) - 1)] @ flip, (w, h))
        thick = max(w // 320, 2)
        cv2.rectangle(
            img,
            (round(x0), round(y0)),
            (round(x0 + fw), round(y0 + fw * h / w)),
            (0, 0, 255),
            thick,
        )
        cv2.putText(
            img,
            f"{n * den / num:.1f} s",
            (w // 50, h // 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            h / 400,
            (0, 0, 255),
            thick,
        )
        tiles.append(
            cv2.resize(img, (width, round(width * h / w)), interpolation=cv2.INTER_AREA)
        )
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack(
        [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    )
    dst = Path(dst)
    # cv2.imwrite fails on Windows paths outside the ANSI code page ("→" in folder names)
    dst.write_bytes(cv2.imencode(dst.suffix or ".jpg", sheet)[1].tobytes())
    return dst
