#!/usr/bin/env bash
# Build the Sphinx documentation and serve it at http://localhost:8000
# Stop the server with Ctrl+C.

set -e
cd "$(dirname "$0")/.."

echo "=== Building docs ==="
if ! sphinx-build docs/ docs/_build/; then
    echo
    echo "Docs build failed. If sphinx is missing, install the docs extras:"
    echo "    pip install -e \".[docs]\""
    exit 1
fi

echo
echo "=== Serving docs at http://localhost:8000 ==="
echo "Press Ctrl+C to stop."
python3 -m http.server -d docs/_build 8000
