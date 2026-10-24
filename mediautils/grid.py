"""Bar grid of a song, and the Shotcut markers used to check it with the user.

The beat detector (:func:`mediautils.timeline_beats`) gives the tempo, not the bars, and
it may lock on the 2 and 4 (strongly marked in blues, shuffle, rock), give one beat out of
two, or miss beats where the drums stop. The method agreed with Fanfan:

1. :func:`numbered_beat_markers`: one numbered marker per detected beat, where there is
   music; the user says which markers are a 1, and which bar the first one is;
2. :class:`BarGrid` labels every beat ``bar.beat``; :func:`bar_markers` writes them
   (the 1s in red, the other beats discreet) for the user to give the structure;
3. :func:`bar_markers` again with the sections (in green), validated before any montage.
"""

import os

import numpy as np

from mediautils.montage import add_markers

RED, GREY, GREEN = "#ff3030", "#5a6a80", "#30d050"


def drum_onsets(
    drums: str | os.PathLike,
    offset: float = 0.0,
    threshold: float = 6.0,
    step: float = 0.01,
) -> np.ndarray:
    """Attacks of a drums stem (e.g. from Demucs): peaks of the rise of its level.

    Parameters
    ----------
    drums : str | os.PathLike
        Drums stem of a rush.
    offset : float
        Time of the rush = time of the project + `offset` (seconds).
    threshold : float
        Minimum rise of the level between two windows (dB).
    step : float
        Window length (seconds).

    Returns
    -------
    numpy.ndarray
        Times of the attacks, in project time.
    """
    import soundfile as sf

    x, sr = sf.read(drums, always_2d=True)
    x = x.mean(axis=1)
    h = int(sr * step)
    n = len(x) // h
    e = 10 * np.log10(np.mean(x[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12)
    flux = np.maximum(np.diff(e), 0)
    k = np.arange(1, len(flux) - 1)
    peaks = k[
        (flux[k] > threshold) & (flux[k] >= flux[k - 1]) & (flux[k] >= flux[k + 1])
    ]
    return (peaks + 1) * step - offset


def subdivide(beats, n: int = 2) -> np.ndarray:
    """Beats between the detected ones, when the detector gives one beat out of `n`.

    >>> subdivide([0.0, 1.0, 2.2]).round(2).tolist()
    [0.0, 0.5, 1.0, 1.6, 2.2]
    """
    b = np.asarray(beats, float)
    steps = np.arange(n) / n
    inner = (b[:-1, None] + steps * np.diff(b)[:, None]).ravel()
    return np.append(inner, b[-1])


def fill_beats(beats, window: int = 8, tolerance: float = 0.35) -> np.ndarray:
    """Beats missed by the detector (e.g. where the drums stop), at equal intervals.

    A gap close to ``k`` times the local beat (median of the `window` intervals before
    and after it) gets ``k - 1`` beats.

    >>> fill_beats([0.0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0]).tolist()
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    """
    b = np.asarray(beats, float)
    d = np.diff(b)
    out = [b[0]]
    for i, gap in enumerate(d):
        near = np.concatenate([d[max(i - window, 0) : i], d[i + 1 : i + 1 + window]])
        local = float(np.median(near)) if len(near) else gap
        k = round(gap / local)
        if k >= 2 and abs(gap / local - k) < tolerance:
            out.extend(b[i] + gap * np.arange(1, k) / k)
        out.append(b[i + 1])
    return np.array(out)


def snap(times, onsets, reach: float = 0.06) -> np.ndarray:
    """Each time moved onto the nearest onset within `reach` seconds (else kept).

    >>> snap([1.0, 2.0], [1.03, 2.5]).tolist()
    [1.03, 2.0]
    """
    t = np.asarray(times, float).copy()
    o = np.sort(np.asarray(onsets, float))
    if not len(o):
        return t
    for i, x in enumerate(t):
        j = np.searchsorted(o, x)
        cand = o[max(j - 1, 0) : j + 1]
        best = cand[np.argmin(abs(cand - x))]
        if abs(best - x) <= reach:
            t[i] = best
    return t


def regular_beats(beats, start: float, window: float = 6.0) -> np.ndarray:
    """Beats redrawn from `start` at the local median beat (±`window` s), forwards and back.

    Over the span of the detected beats; keeps their slow tempo changes, not their jitter.

    >>> regular_beats([0.0, 0.53, 1.0, 1.47, 2.0, 2.5, 3.0], 1.0).round(2).tolist()
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    """
    b = np.asarray(beats, float)

    def period(t):
        near = b[(b > t - window) & (b < t + window)]
        if len(near) < 3:
            near = np.sort(b[np.argsort(abs(b - t))[:8]])
        return float(np.median(np.diff(near)))

    out = [start]
    while out[-1] + period(out[-1]) <= b[-1] + 0.25 * period(out[-1]):
        out.append(out[-1] + period(out[-1]))
    while out[0] - period(out[0]) >= b[0] - 0.25 * period(out[0]):
        out.insert(0, out[0] - period(out[0]))
    return np.array(out)


class BarGrid:
    """Beats of a song labelled with their bar and beat number.

    Parameters
    ----------
    beats : array-like
        Every beat (seconds), regularly spaced (see :func:`subdivide`, :func:`fill_beats`).
    index : int
        Position in `beats` of a beat 1.
    bar : int
        Number of that bar (the bar numbers of the user's structure).
    beats_per_bar : int
        Meter.

    Examples
    --------
        >>> g = BarGrid([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], index=3, bar=2)
        >>> g.label(0)
        (1, 2)
        >>> g.time(2), g.time(2, 3), g.time(2, 1.5)
        (2.0, 3.0, 2.25)
        >>> g.downbeats().tolist()
        [2.0, 4.0]
    """

    def __init__(self, beats, index: int, bar: int = 1, beats_per_bar: int = 4):
        self.beats = np.asarray(beats, float)
        self.per_bar = beats_per_bar
        self.offset = (
            bar - 1
        ) * beats_per_bar - index  # beat number (0 = 1.1) of beats[0]

    @classmethod
    def from_downbeat(
        cls,
        beats,
        time: float,
        bar: int = 1,
        beats_per_bar: int = 4,
        onsets=None,
        reach: float = 0.06,
    ) -> "BarGrid":
        """Grid from the time of a beat 1 given by the user.

        The nearest beat is taken as that 1, and every beat is shifted by the same amount
        so that it falls on `time` (the detector may lag the attacks by a constant, e.g.
        0.24 s on a shuffle); then, with `onsets` (e.g. :func:`drum_onsets`), each beat
        is moved onto the nearest attack within `reach` seconds.
        """
        b = np.asarray(beats, float)
        i = int(np.argmin(abs(b - time)))
        b = b + (time - b[i])
        if onsets is not None:
            b = snap(b, onsets, reach)
        return cls(b, i, bar, beats_per_bar)

    @classmethod
    def between(
        cls,
        beats,
        first: tuple[float, int],
        second: tuple[float, int],
        beats_per_bar: int = 4,
        onsets=None,
        reach: float = 0.06,
        smooth: bool = True,
    ) -> "BarGrid":
        """Grid from two beat 1s given by the user, ``(time, bar)`` each, far apart.

        With `smooth`, the beats are first redrawn from the first 1 at the local median
        beat (:func:`regular_beats`): the detected positions may jitter (a shuffle).
        They are then stretched so that both 1s fall exactly on the user's times (the
        tempo drifts slowly); the number of beats between them must match the number of
        bars. Then, with `onsets`, each beat is moved onto the nearest attack within
        `reach` seconds, without any accumulated error.

        On Roadhouse blues (shuffle, detector on one beat out of two) this gives the
        validated grid within 0.02 s (median), 0.2 s at worst.
        """
        b = np.asarray(beats, float)
        (t1, bar1), (t2, bar2) = first, second
        if smooth:
            b = regular_beats(b, t1)
        i1 = int(np.argmin(abs(b - t1)))
        i2 = i1 + (bar2 - bar1) * beats_per_bar
        beat = float(np.median(np.diff(b)))
        if i2 >= len(b) or abs((b[i2] - t2) - (b[i1] - t1)) > 0.5 * beat:
            raise ValueError(
                "the beats between the two 1s do not match the number of bars (fill_beats first?)"
            )
        b = t1 + (b - b[i1]) * (t2 - t1) / (b[i2] - b[i1])
        if onsets is not None:
            b = snap(b, onsets, reach)
        return cls(b, i1, bar1, beats_per_bar)

    def label(self, i: int) -> tuple[int, int]:
        """``(bar, beat)`` of ``beats[i]``."""
        n = i + self.offset
        return n // self.per_bar + 1, n % self.per_bar + 1

    def time(self, bar: int, beat: float = 1.0) -> float:
        """Time of a beat (a fractional beat is interpolated: 4.5 = the « 4 et »)."""
        k = (bar - 1) * self.per_bar + int(beat) - 1 - self.offset
        frac = beat - int(beat)
        t = self.beats[k]
        return float(t + frac * (self.beats[k + 1] - t)) if frac else float(t)

    def downbeats(self) -> np.ndarray:
        """Times of every beat 1."""
        return np.array([t for i, t in enumerate(self.beats) if self.label(i)[1] == 1])

    def is_downbeat(self, time: float, tolerance: float = 0.1) -> bool:
        """Whether a beat 1 lies within `tolerance` seconds of `time` (to check a user's anchor)."""
        return bool(np.min(abs(self.downbeats() - time)) <= tolerance)


def numbered_beat_markers(path: str | os.PathLike, beats) -> None:
    """One marker per beat, numbered from 1, in a Shotcut project (in place).

    The user answers « markers 43 and 519 are a 1, the first bar starts at marker 3 ».
    """
    add_markers(path, [(float(t), str(k)) for k, t in enumerate(beats, 1)], color=GREY)


def bar_markers(
    path: str | os.PathLike,
    grid: BarGrid,
    sections: dict[int, str] | None = None,
    every_beat: bool = True,
    until: float | None = None,
) -> None:
    """``bar.beat`` markers in a Shotcut project (in place): the 1s in red, the others grey.

    Parameters
    ----------
    path : str | os.PathLike
        The project.
    grid : BarGrid
        The bars.
    sections : dict[int, str], optional
        ``bar -> name``: green markers ``"21.1 Refrain 1"`` on the first beat of each
        section. A section starting before the first beat of the grid (cut bar) gets its
        marker at 0:00.
    every_beat : bool
        Every beat (step 2, to give the structure), or only the 1s (in a montage).
    until : float, optional
        No beat markers after this time (applause, speech after the song).
    """
    sections = dict(sections or {})
    ones, others, greens = [], [], []
    for i, t in enumerate(grid.beats):
        if until is not None and t > until:
            break
        bar, beat = grid.label(i)
        if bar < 1:
            continue
        name = f"{bar}.{beat}"
        if beat == 1 and bar in sections:
            greens.append((float(t), f"{name} {sections.pop(bar)}"))
        elif beat == 1:
            ones.append((float(t), name))
        elif every_beat:
            others.append((float(t), name))
    first_bar = grid.label(0)[0]
    for bar in sorted(sections):
        if bar <= max(first_bar, 1):  # cut at the start of the video
            greens.append((0.0, sections[bar]))
    if others:
        add_markers(path, others, color=GREY)
    if ones:
        add_markers(path, ones, color=RED)
    if greens:
        add_markers(path, sorted(greens), color=GREEN)
