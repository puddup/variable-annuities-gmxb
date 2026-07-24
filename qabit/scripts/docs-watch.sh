#!/usr/bin/env bash
# Build the Sphinx documentation with auto-reload.
# Edits to docs/ or src/qabit/ rebuild automatically and the browser refreshes.
# Stop with Ctrl+C.

set -e
cd "$(dirname "$0")/.."

if ! command -v sphinx-autobuild >/dev/null 2>&1; then
    echo "sphinx-autobuild not found. Install it with:"
    echo "    pip install sphinx-autobuild"
    exit 1
fi

echo "=== Live docs at http://127.0.0.1:8000 ==="
echo "Press Ctrl+C to stop."
sphinx-autobuild docs/ docs/_build/ --watch src/qabit
