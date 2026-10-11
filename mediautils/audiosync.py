"""Synchronization of two recordings of the same event by their soundtracks.

Used to align videos of a concert filmed by several cameras whose clocks
disagree: the offset between the starts of two recordings is the lag that best
matches their soundtracks, found by generalized cross-correlation with phase
transform (GCC-PHAT), which keeps only the timing information of the sound and
is thus robust to different microphones, gains and positions.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def read_audio(path: str | os.PathLike, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read the soundtrack of a media file as a mono signal (requires ffmpeg).

    Parameters
    ----------
    path : str | os.PathLike
        Path to the media file.
    sample_rate : int
        Sampling rate of the returned signal, in Hz.

    Returns
    -------
    numpy.ndarray
        The mono signal, as floats.
    """
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
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
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def audio_offset(
    ref: np.ndarray,
    other: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    center: float | None = None,
    half_window: float | None = None,
) -> tuple[float, float]:
    """Find the offset between two signals of the same event (GCC-PHAT).

    Parameters
    ----------
    ref, other : numpy.ndarray
        The two signals.
    sample_rate : int
        Their sampling rate, in Hz.
    center, half_window : float | None
        If given, only offsets within `half_window` seconds of `center` are
        considered.

    Returns
    -------
    tuple[float, float]
        The offset, in seconds, such that ``other`` at time t matches ``ref``
        at time t + offset, i.e. `other` started `offset` seconds after `ref`
        (negative: before); and a confidence, the height of the best peak
        divided by the best one farther than 50 ms (1 means no clear match).
        Only offsets for which the two signals overlap are considered.

    Examples
    --------
        >>> rng = np.random.default_rng(0)
        >>> event = rng.standard_normal(5 * SAMPLE_RATE)
        >>> ref, other = event[:4 * SAMPLE_RATE], event[SAMPLE_RATE // 2:]
        >>> offset, confidence = audio_offset(ref, other)
        >>> round(offset, 3), confidence > 3
        (0.5, True)
    """
    n = 1 << int(np.ceil(np.log2(len(ref) + len(other))))
    spectrum = np.fft.rfft(ref, n) * np.conj(np.fft.rfft(other, n))
    spectrum /= np.abs(spectrum) + 1e-12
    cc = np.fft.irfft(spectrum, n)
    # Circular lags -> signed lags, keeping only physically possible ones
    # (the two signals overlap), which removes the circular ambiguity.
    lags = np.arange(n)
    lags[lags >= len(ref)] -= n
    valid = (lags > -len(other)) & (lags < len(ref))
    if center is not None:
        valid &= np.abs(lags / sample_rate - center) <= half_window
    values = np.where(valid, cc, -np.inf)
    i = int(np.argmax(values))
    # Parabolic interpolation for sub-sample precision.
    y0, y1, y2 = cc[i - 1], cc[i], cc[(i + 1) % n]
    denominator = y0 - 2 * y1 + y2
    fraction = 0.5 * (y0 - y2) / denominator if denominator else 0.0
    offset = (lags[i] + fraction) / sample_rate
    far = valid & (np.abs(lags - lags[i]) > 0.05 * sample_rate)
    second = np.max(np.where(far, cc, -np.inf)) if far.any() else 0.0
    confidence = float(cc[i] / second) if second > 0 else float("inf")
    return float(offset), confidence


@dataclass
class SyncResult:
    """Result of :func:`sync_videos`.

    Attributes
    ----------
    offset : float
        Start of the other video minus start of the reference, in seconds.
    confidence : float
        Height of the best match relative to the next best (see
        :func:`audio_offset`).
    checks : list[float]
        Offsets found independently on chunks spread over the overlapping
        part, minus `offset`, in seconds: they should all be close to 0.
    """

    offset: float
    confidence: float
    checks: list[float]

    @property
    def consistent(self) -> bool:
        """Whether all the chunk checks agree within one frame at 25 fps."""
        return all(abs(c) <= 0.04 for c in self.checks)


def sync_videos(
    ref: str | os.PathLike,
    other: str | os.PathLike,
    n_checks: int = 3,
    chunk: float = 10.0,
) -> SyncResult:
    """Find the offset between the starts of two videos of the same event.

    Parameters
    ----------
    ref, other : str | os.PathLike
        Paths to the reference video and to the video to place.
    n_checks : int
        Number of chunks of `other` checked independently (within the part
        filmed by both).
    chunk : float
        Length of each checked chunk, in seconds.

    Returns
    -------
    SyncResult
    """
    sr = SAMPLE_RATE
    a, b = read_audio(Path(ref)), read_audio(Path(other))
    offset, confidence = audio_offset(a, b)
    # Part of `other` also covered by `ref`, in seconds of `other`.
    lo = max(0.0, -offset)
    hi = min(len(b) / sr, len(a) / sr - offset)
    checks = []
    if hi - lo >= chunk:
        for k in range(n_checks):
            start = lo + (hi - lo - chunk) * (k + 0.5) / n_checks
            seg = b[int(start * sr) : int((start + chunk) * sr)]
            local, _ = audio_offset(a, seg, center=offset + start, half_window=1.0)
            checks.append(local - start - offset)
    return SyncResult(offset, confidence, checks)
