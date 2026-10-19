import shutil

import numpy as np
import pytest

from mediautils import SyncProject
from tests.test_montage import FPS, make_sync_project, write_wav

pytest.importorskip("scipy")
pytest.importorskip("soundfile")


def test_match_eq():
    from mediautils import match_eq

    rng = np.random.default_rng(0)
    x = rng.standard_normal(48000 * 4)
    fir = match_eq(x, x)
    assert len(fir) == 4095
    assert np.allclose(fir, fir[::-1])  # linear phase
    assert (
        abs(20 * np.log10(np.abs(np.fft.rfft(fir, 8192))[100:3000]).mean()) < 1
    )  # flat when nothing to do


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed")
def test_mix_two_cameras(tmp_path):
    from mediautils import mix_two_cameras

    sr = 48000
    rng = np.random.default_rng(2)
    event = rng.standard_normal(8 * sr) * 0.1
    # reference camera films from 1 s to 7 s, the other one from 0 s to 7 s
    make_sync_project(
        tmp_path / "p.mlt", ref_start=1.0, ref_len=6.0, other_len=7.0, fade=1.0
    )
    write_wav(
        tmp_path / "x_P1000001_son_debruite_Claude.wav",
        np.column_stack([event[sr : 7 * sr]] * 2),
        sr,
    )
    write_wav(
        tmp_path / "y_MVI_0001_son_debruite_Claude.wav", event[: 7 * sr] * 0.5, sr
    )
    P = SyncProject(tmp_path / "p.mlt")
    out = mix_two_cameras(
        P, tmp_path / "mix.wav", target_lufs=-20, switch_in={1.0: 2.0}
    )
    assert out["camera_changes"][0] == round(
        round(1.0 * FPS) / FPS, 2
    )  # (the ends may differ by a few samples)
    assert abs(out["lufs"] + 20) < 0.5
    import soundfile as sf

    y, ysr = sf.read(tmp_path / "mix.wav")
    assert ysr == sr and y.shape == (round(P.video_end / FPS * sr), 2)
    with pytest.raises(FileExistsError):
        mix_two_cameras(P, tmp_path / "mix.wav")
