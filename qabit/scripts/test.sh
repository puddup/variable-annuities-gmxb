#!/usr/bin/env bash
# Run the fast test tier — same as the CI `test` stage.
# Pass extra args through, e.g.:
#     scripts/test.sh -v
#     scripts/test.sh -k put_call_parity
#     scripts/test.sh tests/unit/test_black_scholes.py

set -e
cd "$(dirname "$0")/.."

echo "=== Running tests (excluding slow) ==="
if ! pytest -m "not slow" "$@"; then
    echo
    echo "Tests failed. If pytest is missing, install the test extras:"
    echo "    pip install -e \".[test]\""
    exit 1
fi
