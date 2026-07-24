@echo off
REM Run the fast test tier — same as the CI `test` stage.
REM Pass extra args through, e.g.:
REM     scripts\test.bat -v
REM     scripts\test.bat -k put_call_parity
REM     scripts\test.bat tests\unit\test_black_scholes.py

setlocal
cd /d "%~dp0\.."

echo === Running tests (excluding slow) ===
pytest -m "not slow" %*
if errorlevel 1 (
    echo.
    echo Tests failed. If pytest is missing, install the test extras:
    echo     pip install -e ".[test]"
    exit /b 1
)

endlocal
