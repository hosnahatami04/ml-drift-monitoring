export PYTHONHASHSEED = 0

.PHONY: setup lint format typecheck test cov

setup:
	uv sync --extra dev

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
