# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mediautils** is a Python library for managing photo and video files — updating metadata from standardized file names (and vice-versa), sorting photos by orientation. Scaffolded from [Package Helper 3](https://balouf.github.io/package-helper-3/).

## Build & Development

Uses **uv** as the package/dependency manager and **Hatchling** as the build backend.

```bash
uv sync --all-extras      # Install all deps (including dev group)
uv run pytest              # Run full test suite (includes doctests + coverage)
uv run pytest tests/test_mediautils.py::test_foo   # Run a single test
uv run ruff check .        # Lint
uv run ruff format .       # Format
```

Pytest is configured in `pyproject.toml` with `--doctest-modules` — doctests embedded in source files are executed as part of the test suite. Coverage reports go to `cov/` (HTML) and `coverage.xml`.

## Documentation

Sphinx docs live in `docs/`, built with pydata-sphinx-theme. Source files are MyST Markdown (`.md`). Tutorials are Jupyter notebooks rendered via nbsphinx.

```bash
uv run sphinx-build docs docs/_build/html   # Build docs locally
```

## System Dependencies

- **ffmpeg** (with ffprobe) must be on `PATH` for video operations (`set_time_video`, reading soundtracks for `audiosync`, stream checks).
- **exiftool** must be on `PATH` for RAW image metadata (`set_time_raw`) — the `exif` package only parses JPEG APP1 segments, not TIFF-based RAW formats (e.g. Panasonic `.rw2`) — and for all the video date work (`set_all_dates_video`, `write_precise_start`).
- **numpy** (Python dependency) for `audiosync`.
- Concert montage and sound: extra `concert` (`uv sync --extra concert`: librosa, scipy, soundfile — imported lazily), **melt** (`C:\Program Files\Shotcut\melt.exe`, Shotcut's engine) to render projects, and **uv** to run Demucs (`uv run --with demucs`, pulls PyTorch).

## Architecture

- `mediautils/file_name.py` — file name parsing (datetime/date extraction from standardized and WhatsApp names). `standardize_name()` gives the standardized name and datetime of a file; the naming conventions of the sources met so far are listed in `_KNOWN_FORMATS`: **add one line there for each new source**, so that the scripts handle more and more by themselves (unknown conventions must be refused loudly, never guessed).
- `mediautils/image.py` — image operations (EXIF metadata, orientation detection, resize, batch dispatch)
- `mediautils/video.py` — video metadata: `set_all_dates_video` (every date of a video set to the date of its name, via exiftool, read back and checked), `write_precise_start` (start to the ms and timecode, in XMP), `set_time_video` (older, ffmpeg remux, `creation_time` only), `rotate_video`
- `mediautils/media.py` — dispatchers and batch operations (`set_time`, `process_videos`, `process_whatsapp_files`, `process_date_subdirs`, `process_datetime_filenames`, `process_clock_shift`, `process_timezone_shift`, `gather_yes_no`)
- `mediautils/audiosync.py` — offset between two recordings of the same event from their soundtracks (GCC-PHAT), e.g. to align several cameras of a concert
- `mediautils/shotcut.py` — maintenance of Shotcut projects (`.mlt`): make links relative, follow renamed files
- `mediautils/montage.py` — multi-camera montage of a concert song from a Shotcut synchronization project (one camera per video track; reference camera `"L"` = Lumix, `"W"` = WhatsApp videos, any other `"A"` = Andrea's): `SyncProject` (positions of the sources), `resync_project` (place every source to the frame from the soundtracks, keeping all edit points), `extract_frames` / `contact_sheets` / `timeline_beats` (look at the song), `nervous_shots` + `build_montage` (cuts on the beats, fixed zooms = Shotcut "Size, Position & Rotate"; `blocked` = a camera hidden while someone walks in front of it; falls back on another camera, else black; no reference under a vertical phone video), `trim_project_head` / `trim_project_end`, `use_single_audio`, `export_project` (melt, crf 23 = Shotcut's default quality)
- `mediautils/livesound.py` — sound of a song filmed by two cameras: `denoise_rushes` (Demucs `htdemucs`; the vocals stem, where crowd chatter goes, lowered by 20 dB where nobody sings nor speaks), `mix_two_cameras` (left 75 % A + 25 % L, right the reverse; single-camera parts equalized to the mix colour; quick 0.3 s switch on a beat or a pause), `loudness`, `match_eq`. The module docstring records the trials that failed (heavier models, gating, ms re-alignment)
- `mediautils/audit.py` — audit of a montage before showing it: `montage_shots` (shots of a montage `.mlt`, with the zoom names written by `build_montage`), `vocal_entries`, `audit_cuts` (each cut vs the nearest entry of the voice; shots where the singer sings while not framed), `audit_frames` (3 pictures per shot read from the source files with ffmpeg, on sheets: hidden people, spectators), `audit_headroom` (faces detected with OpenCV in the whole source picture vs each zoom: heads cut, people cut at the edge), `fit_zooms` (zooms defined by the people they show, centred and set in height from the faces of the song: heads ~8 % under the top edge, almost never cut; fit the zooms of every new song with it, the camera framing changes; neighbours may be cut). The structure of a song is a map; the sound and the pictures are the truth. `montage.add_markers`: Shotcut markers, one per section
- `mediautils/framing.py` — framings of a camera as the **intervals of the axis of the musicians**: `Person` (box placed by eye: x0, x1, top of the head, eyes, bottom = hands), `interval_framings` (every interval + a tight shot per person; eyes on the upper third, or up to a quarter; a group's edges do not cut a neighbour in the middle; framings named after the people they really show, near-duplicates merged), `framing_sheet` (the sheet to validate with the user). Replaces `fit_zooms` for new songs
- `mediautils/grid.py` — bar grid of a song and the 3-step marker check agreed with the user: `numbered_beat_markers` (step 1: the user says which markers are a 1), `BarGrid` (`from_downbeat`: one 1, constant shift; `between`: two 1s far apart, beats redrawn at the local median beat by `regular_beats` and stretched between them — for shuffles where the detector jitters or gives one beat out of two), `bar_markers` (step 2: `bar.beat`, 1s red; step 3: sections green; in a montage only the 1s and the sections), helpers `subdivide`, `fill_beats`, `drum_onsets`, `snap` (beats onto the drum attacks, no accumulated error); `count_in` + `band_entry`: the first 1 found alone from the drummer's sticks « 1, 3, 1, 2, 3, 4 » just before the band enters (then no step 1)
- `mediautils/voice.py` — `Voice`: where to cut around the singer from his vocals stem (cut on his breath = the beat before his entry, entries looked for where the structure expects each line, real end of phrase, pauses when he speaks)
- `mediautils/stabilize.py` — `stabilize_video`: copy of a rush filmed on a shaking stand, shifts measured by phase correlation on the still top band of the picture (special case, not a systematic step)
- `mediautils/cli.py` — Click-based CLI entry point (registered as `mediautils` console script)
- `tests/` — pytest tests (also tests the CLI via `click.testing.CliRunner`)

The user runs these functions through `.bat` files (one-liners `uv run --project ... python -c "from mediautils import f; f()"`) placed next to `in`/`out` folders; new features should be callable that way.

## Design Principles

- **No in-place modification**: functions write copies to an output directory, never modify source files. (Low-level helpers documented as modifying a file in place exist, e.g. `write_precise_start`, `relink_project`; they are applied to copies.)
- **Dates of a video** (decided with the user, see the docstrings of `set_all_dates_video` and `process_videos`):
  - every date the file carries takes the date of its name; the name gives the **start** of the recording, in local wall-clock time;
  - only the date tags **already present** are rewritten — never use exiftool `-time:all=`, which creates about a hundred junk tags; QuickTime dates are written in UTC (`-api QuickTimeUTC=1`), DST-aware;
  - dates exiftool cannot write (Lumix: EXIF block, `Panasonic:TimeStamp` stored in UTC, XML clip metadata) are accepted if they give the same wall-clock time ±1 s, or shifted in place by same-length text patching (`_patch_locked_dates`, outside `mdat` only);
  - the XMP Pantry of Adobe exports (metadata of the source files of a montage) is removed; the Adobe save history (`XMP-xmpMM:HistoryWhen`) is the **only** exception kept with its own dates; any other list of dates makes the file refused;
  - a file that cannot be handled is refused and listed, its copy removed, and the batch goes on.
- **Precision beyond the second** (multi-camera sync) goes to XMP only (`xmpDM:shotDate`, `xmpDM:startTimecode`): names and QuickTime dates stay to the second for sorting; no real timecode track (that would need an ffmpeg remux, which drops maker-specific blocks).

## Pitfalls

- exiftool on Windows misreads command-line paths with characters outside the ANSI code page (e.g. "→" in folder names): pass such paths through an argument file (`-charset filename=utf8 -@ file`).
- exiftool `-overwrite_original` rewrites the file and resets its modification time unless `-P` is given.
- ffmpeg cannot write a file onto itself (`-i X ... -y X` fails): write to a temporary file.
- Phones and cameras often record the **end** of a video in its metadata, and old renamings sometimes took that end time: check against the duration before trusting a date.

## CI

GitHub Actions workflows in `.github/workflows/`:
- **build.yml** — tests on Python 3.11–3.14, uploads coverage to Codecov
- **docs.yml** — builds and deploys documentation
- **release.yml** — handles PyPI releases

## Conventions

- Docstrings use **NumPy style** with Sphinx cross-references (`:class:`, `:meth:`, `:attr:`)
- Python target version: 3.11+
- Linter/formatter: Ruff
- Every new public function or class must be re-exported in `mediautils/__init__.py` (pattern: `from mediautils.module import foo as foo`) to allow direct imports (`from mediautils import foo`).
