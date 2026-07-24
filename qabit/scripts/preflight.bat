@echo off
REM Run every check the CI pipeline runs, in the same order.
REM If this passes, your push should pass `lint`, `test`, and `docs`.

setlocal
cd /d "%~dp0\.."

echo === 1/4: ruff check ===
ruff check src\
if errorlevel 1 goto :failed

echo.
echo === 2/4: ruff format --check ===
ruff format --check src\
if errorlevel 1 goto :failed

echo.
echo === 3/4: pytest (fast tier) ===
pytest -m "not slow"
if errorlevel 1 goto :failed

echo.
echo === 4/4: sphinx-build -W ===
sphinx-build -W docs\ docs\_build\
if errorlevel 1 goto :failed

echo.
echo === Preflight passed — safe to push ===
endlocal
exit /b 0

:failed
echo.
echo === Preflight FAILED — fix the errors above before pushing ===
endlocal
exit /b 1
