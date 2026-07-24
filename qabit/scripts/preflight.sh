#!/usr/bin/env bash
# Run every check the CI pipeline runs, in the same order.
# If this passes, your push should pass `lint`, `test`, and `docs`.

cd "$(dirname "$0")/.."

fail() {
    echo
    echo "=== Preflight FAILED — fix the errors above before pushing ==="
    exit 1
}

echo "=== 1/4: ruff check ==="
ruff check src/ || fail

echo
echo "=== 2/4: ruff format --check ==="
ruff format --check src/ || fail

echo
echo "=== 3/4: pytest (fast tier) ==="
pytest -m "not slow" || fail

echo
echo "=== 4/4: sphinx-build -W ==="
sphinx-build -W docs/ docs/_build/ || fail

echo
echo "=== Preflight passed — safe to push ==="
