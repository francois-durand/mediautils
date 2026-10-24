"""Stabilization of a video filmed on a stand that shakes (someone taps the beat on it).

The shift of each picture is measured on still background (e.g. the top band of the
picture, above the heads: ceiling, curtains), by phase correlation against the median
picture, on two zones that must agree (else somebody walks into the band: the shift is
interpolated). Each picture is then moved back; a thin black border appears where the
picture is missing (hidden by any zoom of 1.03 or more). A special case, not a step of
every montage.
"""

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
