#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev

uv run ruff check .
uv run ruff format --check .

uv run pre-commit run --all-files

uv run pytest