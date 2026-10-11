export PYTHONHASHSEED = 0

.PHONY: setup lint format typecheck test cov data train calibrate evaluate evaluate-smoke reproduce

setup:
	uv sync --extra dev

data:
	uv run driftmon download-data

train:
	uv run driftmon train

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term-missing

calibrate:
	uv run driftmon calibrate

evaluate:
	uv run driftmon evaluate

evaluate-smoke:
	uv run driftmon evaluate --smoke

# Two full evaluations from scratch; the results files must be byte-identical.
reproduce:
	rm -rf artifacts/reproduce
	uv run driftmon evaluate --out-dir artifacts/reproduce/run1
	uv run driftmon evaluate --out-dir artifacts/reproduce/run2
	shasum -a 256 artifacts/reproduce/run1/results.json artifacts/reproduce/run2/results.json
	cmp artifacts/reproduce/run1/results.json artifacts/reproduce/run2/results.json
