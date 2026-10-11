"""Tests for `mediautils` package."""

import shutil
import subprocess
from datetime import datetime

import pytest
from click.testing import CliRunner

from mediautils import cli, process_videos


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("exiftool") is None,
    reason="needs ffmpeg and exiftool",
)
def test_process_videos(tmp_path, capsys):
    """Rename an Android video, set all its dates, and complain about the rest."""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=64x64:d=1",
            "-metadata",
            "creation_time=2001-01-01T00:00:00Z",
            str(in_dir / "VID_20190924_162534168.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    shutil.copy(in_dir / "VID_20190924_162534168.mp4", in_dir / "MVI_1234.mp4")
    (in_dir / "Série").mkdir()
    shutil.copy(
        in_dir / "VID_20190924_162534168.mp4", in_dir / "Série" / "20190103_184939.mp4"
    )
    (in_dir / "notes.txt").write_text("not a video")

    results = process_videos(in_dir, out_dir)

    expected = out_dir / "20190924_162534_VID_20190924_162534168.mp4"
    in_subdir = out_dir / "Série" / "20190103_184939.mp4"
    assert sorted(results) == sorted([expected, in_subdir])
    assert sorted(
        p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*.mp4")
    ) == [
        "20190924_162534_VID_20190924_162534168.mp4",
        "Série/20190103_184939.mp4",
    ]
    stored = subprocess.run(
        [
            "exiftool",
            "-s3",
            "-api",
            "QuickTimeUTC=0",
            "-QuickTime:CreateDate",
            "-Track1:MediaCreateDate",
            str(expected),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    # 16:25:34 in Paris during daylight saving time is 14:25:34 UTC.
    assert stored == ["2019:09:24", "14:25:34", "2019:09:24", "14:25:34"]
    assert datetime.fromtimestamp(expected.stat().st_mtime) == datetime(
        2019, 9, 24, 16, 25, 34
    )
    assert datetime.fromtimestamp(in_subdir.stat().st_mtime) == datetime(
        2019, 1, 3, 18, 49, 39
    )
    output = capsys.readouterr().out
    assert "MVI_1234.mp4: unknown file name convention" in output
    assert "notes.txt: not a video extension" in output


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "mediautils.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
