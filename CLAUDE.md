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

- **ffmpeg** must be on `PATH` for video metadata operations (`set_time_video`).

## Architecture

- `mediautils/file_name.py` — file name parsing (datetime/date extraction from standardized and WhatsApp names)
- `mediautils/image.py` — image operations (EXIF metadata, orientation detection, resize, batch dispatch)
- `mediautils/video.py` — video metadata manipulation (MP4 `creation_time` via ffmpeg)
- `mediautils/media.py` — dispatchers and batch operations (`set_time`, `process_wa_files`, `process_directory_files`, `process_standard_files`)
- `mediautils/cli.py` — Click-based CLI entry point (registered as `mediautils` console script)
- `tests/` — pytest tests (also tests the CLI via `click.testing.CliRunner`)

## Design Principles

- **No in-place modification**: functions write copies to an output directory, never modify source files.

## CI

GitHub Actions workflows in `.github/workflows/`:
- **build.yml** — tests on Python 3.11–3.14, uploads coverage to Codecov
- **docs.yml** — builds and deploys documentation
- **release.yml** — handles PyPI releases

## Conventions

- Docstrings use **NumPy style** with Sphinx cross-references (`:class:`, `:meth:`, `:attr:`)
- Python target version: 3.11+
- Linter/formatter: Ruff
