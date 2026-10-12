export PYTHONHASHSEED = 0

.PHONY: setup lint format typecheck test cov data train calibrate evaluate evaluate-smoke kpi-gate reproduce dashboard audit

setup:
	uv sync --extra dev --extra dashboard

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
	uv run pytest --cov --cov-report=term-missing --cov-fail-under=85

calibrate:
	uv run driftmon calibrate

evaluate:
	uv run driftmon evaluate

evaluate-smoke:
	uv run driftmon evaluate --smoke

kpi-gate:
	uv run driftmon kpi-gate

dashboard:
	uv run driftmon dashboard

# Two full evaluations from scratch; the results files must be byte-identical.
reproduce:
	rm -rf artifacts/reproduce
	uv run driftmon evaluate --out-dir artifacts/reproduce/run1
	uv run driftmon evaluate --out-dir artifacts/reproduce/run2
	shasum -a 256 artifacts/reproduce/run1/results.json artifacts/reproduce/run2/results.json
	cmp artifacts/reproduce/run1/results.json artifacts/reproduce/run2/results.json

# Needs network: pip-audit queries the vulnerability database for every
# locked package.
audit:
	mkdir -p artifacts
	uv export --frozen --all-extras --no-hashes --no-emit-project --format requirements-txt -o artifacts/requirements-audit.txt
	uvx pip-audit==2.9.0 --requirement artifacts/requirements-audit.txt --no-deps --disable-pip
