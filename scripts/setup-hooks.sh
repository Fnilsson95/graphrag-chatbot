#!/bin/sh

set -e

# Use the shared hooks in githook/.
git config core.hooksPath githooks

echo "Git hooks path configured to ./githooks"
echo "Shared commit-msg hook is now active."
echo "Shared pre-commit hook is now active (Ruff runs via pre-commit)."

# Check that uv is available
if command -v uv >/dev/null 2>&1; then
  echo "uv is installed."
  echo "Run this once to install project dependencies:"
  echo "  uv sync"
else
  echo "uv is not installed."
  echo "Install uv first, then run:"
  echo "  uv sync"
fi

