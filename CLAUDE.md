# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mediautils** is a Python library for managing photo and video files — updating metadata from standardized file names (and vice-versa), sorting photos by orientation. It is scaffolded from the [Package Helper 3](https://balouf.github.io/package-helper-3/) Cookiecutter template. Currently at v0.1.0 with placeholder classes; the real media-utils logic is being built out.

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

## Architecture

- `mediautils/` — main package, publicly exports `MyClass1`, `MyClass2`, `MyClass3` from `__init__.py`
- `mediautils/cli.py` — Click-based CLI entry point (registered as `mediautils` console script)
- `mediautils/sub_package_1/` and `sub_package_2/` — subpackages containing the class implementations
- `tests/` — pytest tests (also tests the CLI via `click.testing.CliRunner`)
- `legacy_notebooks/` — old Jupyter notebooks (not part of the package)

## CI

GitHub Actions workflows in `.github/workflows/`:
- **build.yml** — tests on Python 3.11–3.14, uploads coverage to Codecov
- **docs.yml** — builds and deploys documentation
- **release.yml** — handles PyPI releases

## Conventions

- Docstrings use **NumPy style** with Sphinx cross-references (`:class:`, `:meth:`, `:attr:`)
- Python target version: 3.11+
- Linter/formatter: Ruff
