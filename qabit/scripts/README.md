# Helper scripts

Convenience wrappers for the common dev workflows. Use the `.bat` files on
Windows, the `.sh` files on macOS or Linux.

Run them from anywhere — they `cd` to the project root automatically.

| Script | What it does |
|---|---|
| `docs` | Build docs and serve at <http://localhost:8000> |
| `docs-watch` | Auto-rebuild on save (needs `pip install sphinx-autobuild`) |
| `test` | Run the fast test tier — same as CI's `test` stage. Extra args pass through to pytest. |
| `preflight` | Run lint + format-check + tests + strict docs build. If this passes, your push will pass CI. |

## First-time setup

From the project root:

```
pip install -e ".[test,docs]"
```

That installs the library in editable mode plus the test and docs dependencies.

## Examples

Windows:

```
scripts\docs.bat
scripts\test.bat -v
scripts\test.bat -k put_call_parity
scripts\preflight.bat
```

macOS / Linux:

```
./scripts/docs.sh
./scripts/test.sh -v
./scripts/test.sh -k put_call_parity
./scripts/preflight.sh
```
