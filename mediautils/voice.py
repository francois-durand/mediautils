"""Where to cut around the singer: his entries, his breaths, the ends of his phrases.

Read from the vocals stem of a rush (e.g. Demucs). Rules agreed with Fanfan:

- a cut towards the singer is made **on his breath**, the beat before his entry (often
  the 4 of the bar before), not on the 1 nor just before his first note;
- his entries are looked for **where the structure expects each line** (he often starts
  on an upbeat), not blindly over the whole song;
- a cut away from him waits for the **real end of his phrase** (voice silent a moment),
  on the next beat; never a cut in the middle of a phrase, interjections included;
- when he speaks (presentation, joke), cut in a **pause** of his speech.
"""

import os
from functools import cached_property

import numpy as np

from mediautils.audit import vocal_entries


class Voice:
    """Level of the voice of a rush, in project time.

    Parameters
    ----------
    vocals : str | os.PathLike
        Vocals stem of the rush.
    offset : float
        Time of the rush = time of the project + `offset` (the in-point of the rush in
        the project, in seconds).
    step : float
        Window length (seconds).
    silent, loud : float
        Levels (dB) below which the voice is silent, above which it sings.
    """

    def __init__(
        self,
        vocals: str | os.PathLike,
        offset: float = 0.0,
        step: float = 0.05,
        silent: float = -40.0,
        loud: float = -34.0,
    ):
        import soundfile as sf

        self.vocals, self.offset, self.step = vocals, offset, step
        self.silent, self.loud = silent, loud
        x, sr = sf.read(vocals, always_2d=True)
        x = x.mean(axis=1)
        h = int(sr * step)
        n = len(x) // h
        self.db = 10 * np.log10(np.mean(x[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12)

    def _i(self, t: float) -> int:
        return int((t + self.offset) / self.step)

    def _t(self, i: int) -> float:
        return i * self.step - self.offset

    def level(self, t: float) -> float:
        """Level of the voice at `t` (dB)."""
        return float(self.db[min(max(self._i(t), 0), len(self.db) - 1)])

    def sings(self, a: float, b: float) -> float:
        """Share of the time between `a` and `b` where the voice is loud."""
        i, j = self._i(a), self._i(b)
        return float((self.db[i:j] > self.loud).mean()) if j > i else 0.0

    @cached_property
    def entries(self) -> np.ndarray:
        """Entries of the voice (after 0.4 s of silence), in project time."""
        return vocal_entries(self.vocals, self.offset, silence=0.4)

    def entry(self, a: float, b: float) -> float | None:
        """First entry between `a` and `b` (the window where the structure expects a line)."""
        e = self.entries[(self.entries >= a) & (self.entries < b)]
        return float(e[0]) if len(e) else None

    def breath_cut(self, t: float, beats, reach: float = 0.8) -> float:
        """Cut for an entry at `t`: on the beat before it, if the voice is silent there.

        Otherwise just before the entry (`beats`: every beat of the song, seconds).
        """
        b = np.asarray(beats, float)
        k = np.searchsorted(b, t - 0.1) - 1
        if k >= 0:
            q = float(b[k])
            i = self._i(q)
            if t - q < reach and self.db[max(i - 1, 0) : i + 2].min() < self.silent:
                return q
        return t - 0.05

    def line(self, a: float, b: float, beats, default: float) -> float:
        """Cut for a line expected between `a` and `b`: its breath, else `default` (the 1)."""
        e = self.entry(a, b)
        return self.breath_cut(e, beats) if e is not None else default

    def phrase_end(self, t: float, beats, quiet: float = 0.4) -> float:
        """End of the phrase going on at `t`: the beat after the voice is silent `quiet` s."""
        b = np.asarray(beats, float)
        i, k = self._i(t), int(round(quiet / self.step))
        while i < len(self.db) - k and not (self.db[i : i + k] < self.silent).all():
            i += 1
        j = min(np.searchsorted(b, self._t(i)), len(b) - 1)
        return float(b[j])

    def pauses(
        self, a: float, b: float, min_length: float = 0.15
    ) -> list[tuple[float, float]]:
        """Pauses of the voice (silent at least `min_length` s) between `a` and `b`."""
        i0, i1 = max(self._i(a), 0), min(self._i(b), len(self.db))
        quiet = self.db[i0:i1] < self.silent
        out, k, need = [], 0, int(round(min_length / self.step))
        while k < len(quiet):
            if quiet[k]:
                j = k
                while j < len(quiet) and quiet[j]:
                    j += 1
                if j - k >= need:
                    out.append((self._t(i0 + k), self._t(i0 + j)))
                k = j
            else:
                k += 1
        return out

    def longest_pause(self, a: float, b: float) -> float | None:
        """Middle of the longest pause between `a` and `b` (to cut while he speaks)."""
        p = self.pauses(a, b, min_length=self.step)
        if not p:
            return None
        s, e = max(p, key=lambda x: x[1] - x[0])
        return (s + e) / 2
