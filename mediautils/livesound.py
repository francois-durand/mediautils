"""Sound of a concert song filmed by two cameras: crowd-noise reduction and mix.

Made for concerts recorded only by the microphones of two cameras (no mixing
desk): a reference camera ``"L"`` (the Lumix, stereo, the better sound) and
another one ``"A"`` (Andrea's Canon, mono), placed on both sides of the stage.
Positions come from a synchronization project (:class:`~mediautils.montage.SyncProject`).
Other sources of the project (e.g. ``"W"``, a WhatsApp video) are not used for the sound.

- :func:`denoise_rushes`: the soundtrack of each camera is separated by Demucs
  (``htdemucs``) into vocals / drums / bass / other; the *vocals* stem is lowered by
  20 dB where nobody sings nor speaks (the crowd chatter goes there), with fades.
  Writes ``<rush>_son_<tag>_Claude.wav`` next to each rush.
- :func:`mix_two_cameras`: a single stereo sound for the whole timeline: left =
  75 % A + 25 % L, right = 25 % A + 75 % L (Fanfan's mix), both cameras at the same
  loudness; where a single camera films, its sound on both sides, equalized to the
  colour of the mix; the passage to the mix is quick (0.3 s) and placed by the
  caller on a beat or in a pause (a long fade from camera A is heard as a muffled
  bass: its microphone renders the bass weakly).

Lessons kept from the trials (Burlingue concert, 2026-10): heavier separation
models (``htdemucs_6s``, ``htdemucs_ft``, RoFormer) were not better enough for their
cost; gating the vocals stem by level, or by pitch, cannot tell the singer from the
crowd (the singer is only 2-4 dB above it, and guitars leak into the stem);
re-aligning the cameras to the millisecond around cross-fades is not reliable
(the offset depends on the dominant source). Requires ffmpeg, the extra
``concert`` (soundfile, scipy) and ``uv`` (Demucs is run with ``uv run --with demucs``).
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 48000
DEMUCS = [
    "uv",
    "run",
    "--with",
    "demucs",
    "--with",
    "soundfile",
    "python",
    "-m",
    "demucs",
    "-n",
    "htdemucs",
]


def _gain_curve(t, intervals, floor, fade):
    g = np.ones_like(t)
    for iv in intervals:
        a, b, f_in, f_out = iv if len(iv) == 4 else (*iv, fade, fade)
        depth = np.minimum(
            np.clip((t - a) / f_in, 0, 1), np.clip((b - t) / f_out, 0, 1)
        )
        g = np.minimum(g, 1 - depth * (1 - floor))
    return g


def denoise_rushes(
    project,
    work_dir: str | os.PathLike,
    quiet: list[tuple],
    floor: float = 0.1,
    fade: float = 1.0,
    tag: str = "debruite",
    demucs: list[str] = DEMUCS,
) -> list[tuple[str, float]]:
    """Write a denoised soundtrack next to each rush of a synchronization project.

    Parameters
    ----------
    project : mediautils.montage.SyncProject
        The synchronization project (gives the rushes and their positions).
    work_dir : str | os.PathLike
        Folder for intermediate files (extracted sound, Demucs stems; reused if present).
    quiet : list[tuple]
        Parts of the timeline where nobody sings nor speaks: ``(a, b)`` or
        ``(a, b, fade_in, fade_out)`` in seconds (``a < 0``: from the very start).
        Rule of thumb (Fanfan): at the start of a song, fade in over 2 s; between
        sung and instrumental parts, 1 s is fine.
    floor : float
        Gain of the vocals stem in these parts (0.1 = -20 dB).
    fade : float
        Default fade length, in seconds.
    tag : str
        The files are named ``<rush>_son_<tag>_Claude.wav`` (a new tag avoids
        touching files locked by Shotcut).
    demucs : list[str]
        Command running Demucs (the input files are appended).

    Returns
    -------
    list[tuple[str, float]]
        Name of each file written and the fraction of the rush where the vocals are lowered.
    """
    import soundfile as sf

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rushes = project.rushes()
    wavs = []
    for name, _ in rushes:
        w = work / (Path(name).stem + ".wav")
        if not w.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(project.dir / name),
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    str(w),
                ],
                check=True,
            )
        wavs.append(w)
    todo = [
        w
        for w in wavs
        if not (work / "stems" / "htdemucs" / w.stem / "vocals.wav").exists()
    ]
    if todo:
        subprocess.run(
            [*demucs, "-o", str(work / "stems"), *map(str, todo)],
            check=True,
            capture_output=True,
        )
    out = []
    for (name, t0), w in zip(rushes, wavs):
        d = work / "stems" / "htdemucs" / w.stem
        voc, sr = sf.read(d / "vocals.wav")
        inst = sum(sf.read(d / f"{s}.wav")[0] for s in ("drums", "bass", "other"))
        t = np.arange(len(voc)) / sr + t0 / project.fps
        g = _gain_curve(t, quiet, floor, fade)
        tmp = work / f"{w.stem}_{tag}_44k.wav"
        sf.write(tmp, inst + voc * g[:, None], sr, subtype="FLOAT")
        dur = float(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(project.dir / name),
                ],
                capture_output=True,
                text=True,
            ).stdout
        )
        dst = project.dir / (Path(name).stem + f"_son_{tag}_Claude.wav")
        if dst.exists():  # keep the previous versions: _v1, _v2...
            k = 1
            while dst.with_name(f"{dst.stem}_v{k}.wav").exists():
                k += 1
            dst.rename(dst.with_name(f"{dst.stem}_v{k}.wav"))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(tmp),
                "-af",
                f"aresample={SR},apad=whole_dur={dur}",
                "-t",
                f"{dur}",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            check=True,
        )
        out.append((dst.name, float((g < 0.99).mean())))
    return out


def loudness(x: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness (LUFS, EBU R128) of a signal, measured by ffmpeg."""
    import soundfile as sf
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.wav"
        sf.write(p, x, sr, subtype="FLOAT")
        err = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(p),
                "-af",
                "ebur128",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        ).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def match_eq(
    src: np.ndarray,
    target: np.ndarray,
    sr: int = SR,
    max_db: float = 9.0,
    taps: int = 4095,
) -> np.ndarray:
    """Linear-phase FIR giving `src` the long-term spectral shape of `target` (both mono).

    The correction is smoothed (about a third of an octave), limited to
    ``±max_db`` and normalized (median 0 dB): it changes the colour, not the level.
    Apply it with ``scipy.signal.fftconvolve(x, fir, mode="same")`` (no delay).
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import firwin2, welch

    f, ps = welch(src, sr, nperseg=8192)
    _, pt = welch(target, sr, nperseg=8192)
    g = 10 * np.log10((pt + 1e-12) / (ps + 1e-12))
    grid = np.arange(np.log2(20), np.log2(sr / 2), 1 / 24)
    gg = uniform_filter1d(np.interp(grid, np.log2(np.maximum(f, 20)), g), size=8)
    gg = np.clip(gg - np.median(gg), -max_db, max_db)
    freqs = np.concatenate([[0], 2**grid, [sr / 2]]) / (sr / 2)
    gains = 10 ** (np.concatenate([[gg[0]], gg, [gg[-1]]]) / 20)
    freqs, idx = np.unique(np.clip(freqs, 0, 1), return_index=True)
    return firwin2(taps, freqs, gains[idx])


def _apply(fir, x):
    from scipy.signal import fftconvolve

    return np.stack(
        [fftconvolve(x[:, c], fir, mode="same") for c in range(x.shape[1])], axis=1
    )


def mix_two_cameras(
    project,
    dst: str | os.PathLike,
    target_lufs: float = -19.6,
    share: float = 0.75,
    ramp_in: float = 0.3,
    ramp_out: float = 3.0,
    switch_in: dict[float, float] | None = None,
    fade_in_at: tuple[float, float] | None = None,
    end_s: float | None = None,
    tag: str = "debruite",
) -> dict:
    """Write the sound of a song from the denoised soundtracks of the two cameras.

    With the reference camera only (no ``"A"`` rush), its own stereo sound, at the target loudness.

    Parameters
    ----------
    project : mediautils.montage.SyncProject
        The synchronization project (positions of the rushes).
    dst : str | os.PathLike
        The stereo WAV written, covering the timeline from 0 (must not exist).
    target_lufs : float
        Loudness of the result (-19.6 LUFS: the level of Fanfan's previous mixes).
    share : float
        Weight of the dominant camera on each side (A on the left, L on the right).
    ramp_in, ramp_out : float
        Length (s) of the passage to the mix when a camera starts, and from the
        mix when one stops.
    switch_in : dict[float, float] | None
        ``{time a camera starts (approx.): time the passage starts}``: the camera
        filming alone goes on until then (choose a beat, a hit or a pause).
    fade_in_at : tuple[float, float] | None
        ``(start, duration)``: silence before `start`, then a fade-in (for a project
        whose head is cut by :func:`~mediautils.montage.trim_project_head`).
    end_s : float | None
        Length of the result (default: up to the last frame filmed by a camera).
    tag : str
        Tag of the denoised files (see :func:`denoise_rushes`).

    Returns
    -------
    dict
        Loudness and peak of the result, times where a camera starts or stops.
    """
    import soundfile as sf

    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    switch_in = switch_in or {}
    P = project
    # by default, up to the last frame filmed by a camera (an audio track of the project may be longer)
    n = int(round((end_s if end_s is not None else P.video_end / P.fps) * SR))
    tracks = {}
    for name, t0 in P.rushes():
        x, sr = sf.read(
            P.dir / (Path(name).stem + f"_son_{tag}_Claude.wav"), always_2d=True
        )
        if sr != SR:
            raise ValueError(f"{name}: {sr} Hz")
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        y = np.zeros((n, 2))
        off = int(round(t0 / P.fps * SR))
        a, b = max(0, off), min(n, off + len(x))
        y[a:b] = x[a - off : b - off]
        present = np.zeros(n, bool)
        present[a:b] = True
        cam = P.camera(name)
        if cam in tracks:
            y0, p0 = tracks[cam]
            y, present = np.where(p0[:, None], y0, y), p0 | present
        tracks[cam] = (y, present)
    t = np.arange(n) / SR
    if "A" not in tracks:  # a single camera (the Lumix): its own stereo sound
        return _finish(tracks["L"][0], t, [], target_lufs, fade_in_at, dst)
    (A, pA), (L, pL) = tracks["A"], tracks["L"]
    A = np.repeat(A.mean(axis=1, keepdims=True), 2, axis=1)  # camera A records mono
    both = pA & pL
    A = A * 10 ** ((target_lufs - loudness(A[both])) / 20)
    L = L * 10 ** ((target_lufs - loudness(L[both])) / 20)
    blend = np.column_stack(
        [
            share * A[:, 0] + (1 - share) * L[:, 0],
            (1 - share) * A[:, 1] + share * L[:, 1],
        ]
    )
    ref = blend[both].mean(axis=1)
    A1 = _apply(match_eq(A[both, 0], ref), A)
    L1 = _apply(match_eq(L[both].mean(axis=1), ref), L)
    lb = loudness(blend[both])
    A1 *= 10 ** ((lb - loudness(A1[both])) / 20)
    L1 *= 10 ** ((lb - loudness(L1[both])) / 20)
    w = both.astype(float)
    single = np.where((pA & ~pL)[:, None], A1, L1)
    edges = np.flatnonzero(np.diff(both.astype(int)))
    for e in edges:
        te = (e + 1) / SR
        if both[e + 1]:
            start = next((s for k, s in switch_in.items() if abs(k - te) < 1.0), te)
            alone = A1 if pA[e] else L1
            m = (t >= te) & (t < start)
            w[m] = 0.0
            single[m] = alone[m]
            m = (t >= start) & (t < start + ramp_in)
            w[m] = (t[m] - start) / ramp_in
            single[m] = alone[m]
        else:
            m = (t >= te - ramp_out) & (t < te)
            w[m] = np.minimum(w[m], (te - t[m]) / ramp_out)
            single[m] = A1[m] if pA[e + 1] else L1[m]
    out = w[:, None] * blend + (1 - w[:, None]) * single
    return _finish(out, t, edges, target_lufs, fade_in_at, dst)


def _finish(out, t, edges, target_lufs, fade_in_at, dst):
    """Fade-in, loudness, peak limit, write; the summary returned by :func:`mix_two_cameras`."""
    import soundfile as sf

    out = np.array(out, dtype=float)
    part = np.ones(len(t), bool)
    if fade_in_at is not None:
        a, d = fade_in_at
        out *= np.clip((t - a) / d, 0, 1)[:, None]
        part = t >= a
    out *= 10 ** ((target_lufs - loudness(out[part])) / 20)
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out *= 0.98 / peak
    sf.write(dst, out, SR, subtype="PCM_16")
    return {
        "lufs": loudness(out[part]),
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(out)))),
        "camera_changes": [round(e / SR, 2) for e in edges],
    }
