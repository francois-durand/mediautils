"""Tests for `mediautils` package."""

import os
import shutil
import struct
import subprocess
from datetime import datetime, timedelta

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


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("exiftool") is None,
    reason="needs ffmpeg and exiftool",
)
def test_date_lists(tmp_path, capsys):
    """The Adobe save history is kept as is; any other list of dates is refused."""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    base = in_dir / "base.mp4"
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "color=black:s=64x64:d=1", str(base)],
        check=True,
        capture_output=True,
    )
    history = in_dir / "20260208_235001_history.mp4"
    other = in_dir / "20260208_235002_other.mp4"
    for path, tag in ((history, "HistoryWhen"), (other, "VersionsModifyDate")):
        shutil.copy(base, path)
        subprocess.run(
            [
                "exiftool",
                "-overwrite_original",
                f"-XMP-xmpMM:{tag}+=2026:02:16 13:10:27",
                f"-XMP-xmpMM:{tag}+=2026:02:16 13:10:44",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    base.unlink()

    results = process_videos(in_dir, out_dir)

    assert [p.name for p in results] == [history.name]
    kept = subprocess.run(
        ["exiftool", "-s3", "-XMP-xmpMM:HistoryWhen", str(results[0])],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "2026:02:16 13:10:27" in kept and "2026:02:16 13:10:44" in kept
    assert "lists of dates not handled" in capsys.readouterr().out
    assert not (out_dir / other.name).exists()


def _atom(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def test_patch_locked_dates(tmp_path):
    """Locked dates are shifted in metadata atoms only, with the same length."""
    from mediautils.video import _patch_locked_dates

    exif, xml = b"2026:04:12 17:07:44", b"2026-04-12T17:07:45+01:00"
    moov = _atom(b"moov", b"x" + exif + b"\x00" + exif + b"\x00")
    meta = _atom(b"meta", b"<StartDate>" + xml + b"</StartDate>")
    mdat = _atom(b"mdat", b"video" + exif + b"data")  # must not be touched
    path = tmp_path / "v.mp4"
    path.write_bytes(_atom(b"ftyp", b"isom") + moov + meta + mdat)
    tags = {
        "ExifIFD:DateTimeOriginal": "2026:04:12 17:07:44",
        "ExifIFD:CreateDate": "2026:04:12 17:07:44",
        "XML:StartDate": "2026:04:12 17:07:45+01:00",
    }

    _patch_locked_dates(path, tags, timedelta(minutes=-15))

    data = path.read_bytes()
    assert data.count(b"2026:04:12 16:52:44") == 2
    assert b"2026-04-12T16:52:45+01:00" in data
    assert data.endswith(mdat)

    # A date found a different number of times than expected: nothing changes.
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="found 2 time"):
        _patch_locked_dates(
            path, {"ExifIFD:CreateDate": "2026:04:12 16:52:44"}, timedelta(hours=1)
        )
    assert path.read_bytes() == before


def test_relink_project(tmp_path):
    """Links become bare names of files next to the project; nothing else changes."""
    from mediautils import project_files, relink_project

    (tmp_path / "20250405_205522_P1045670.MP4").write_bytes(b"")
    (tmp_path / "20250405_205454_MVI_0935_from_Andrea.MOV").write_bytes(b"")
    old_andrea = "20250405_231748_MVI_0935_from_Andrea.MOV"
    project = tmp_path / "p.mlt"
    original = (
        '<mlt>\n<chain id="chain0" out="00:04:07.200">\n'
        '<property name="resource">20250405_205522_P1045670.MP4</property>\n'
        '<property name="shotcut:caption">20250405_205522_P1045670.MP4</property>\n'
        "</chain>\n"
        '<chain id="chain1">\n'
        f'<property name="resource">c:/Dropbox/A/old place/01 Cocaine/{old_andrea}</property>\n'
        f'<property name="shotcut:caption">{old_andrea}</property>\n'
        "</chain>\n"
        '<property name="resource">c:/Dropbox/B/music.mp3</property>\n'
        '<entry producer="chain1" in="00:00:01.101" out="00:03:46.530"/>\n</mlt>\n'
    )
    project.write_bytes(original.encode("utf-8"))

    changes = relink_project(
        project, renames={old_andrea: "20250405_205454_MVI_0935_from_Andrea.MOV"}
    )

    assert len(changes) == 2  # Andrea's resource and caption
    assert project_files(project) == [
        "20250405_205522_P1045670.MP4",
        "20250405_205454_MVI_0935_from_Andrea.MOV",
        "c:/Dropbox/B/music.mp3",  # not next to the project: left as is
    ]
    assert 'in="00:00:01.101" out="00:03:46.530"' in project.read_text("utf-8")
    assert (tmp_path / "p.mlt.bak").read_text("utf-8") == original


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("exiftool") is None,
    reason="needs ffmpeg and exiftool",
)
def test_write_precise_start(tmp_path):
    """The start is recorded in XMP to the millisecond, with a timecode."""
    from mediautils import write_precise_start

    path = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "color=black:s=64x64:d=1:r=25", str(path)],
        check=True,
        capture_output=True,
    )
    mtime = datetime(2025, 4, 5, 20, 55, 22).timestamp()
    os.utime(path, (mtime, mtime))
    write_precise_start(path, datetime(2025, 4, 5, 20, 55, 22, 668000), frame_rate=25)
    assert path.stat().st_mtime == mtime  # file times are kept
    out = subprocess.run(
        [
            "exiftool",
            "-s3",
            "-n",
            "-XMP-xmpDM:ShotDate",
            "-XMP-xmpDM:StartTimecodeTimeFormat",
            "-XMP-xmpDM:StartTimecodeTimeValue",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    assert out[:3] == ["2025:04:05 20:55:22.668+02:00", "25Timecode", "20:55:22:17"]


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "mediautils.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
