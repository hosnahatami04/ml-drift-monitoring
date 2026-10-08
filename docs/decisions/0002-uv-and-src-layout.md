# 0002. Use uv with a src layout

Date: 2026-10-08

## Status

Accepted

## Context

The evaluation results in this repository are only credible if anyone can
reproduce them. That requires fully pinned dependencies and an installation
that behaves the same locally, in CI, and inside Docker. The package must also
be importable only when properly installed, so tests exercise the installed
package and never a stray module picked up from the repository root.

## Decision

- Dependencies are managed with uv: versions resolved in `pyproject.toml` and
  pinned in `uv.lock`, installed with `uv sync --extra dev`. The dev
  dependencies are declared as a standard optional extra, so
  `pip install -e ".[dev]"` remains a supported fallback.
- The package lives under `src/driftmon` (src layout). Imports resolve to the
  installed package, not to the working directory.
- Python is pinned to 3.11 via `.python-version` and CI.

## Alternatives considered

- pip + requirements.txt: no built-in lockfile with hashes across platforms;
  resolution drift between machines.
- Poetry: works, but slower resolution and a second lockfile format while uv
  already covers Python pinning, virtualenvs, and locking in one tool.
- Flat layout: simpler paths, but allows accidental imports of the
  un-installed package from the repository root, which hides packaging bugs.

## Consequences

Contributors need uv installed (or must use the pip fallback). All Makefile
targets run tools through `uv run`, so there is no activation step. Lockfile
updates are explicit diffs reviewed like code.
