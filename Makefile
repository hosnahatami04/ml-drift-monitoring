export PYTHONHASHSEED = 0

.PHONY: setup lint format typecheck test cov data train

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
