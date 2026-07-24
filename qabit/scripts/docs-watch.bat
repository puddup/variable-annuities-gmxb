@echo off
REM Build the Sphinx documentation with auto-reload.
REM Edits to docs\ or src\qabit\ rebuild automatically and the browser refreshes.
REM Stop with Ctrl+C.

setlocal
cd /d "%~dp0\.."

where sphinx-autobuild >nul 2>&1
if errorlevel 1 (
    echo sphinx-autobuild not found. Install it with:
    echo     pip install sphinx-autobuild
    exit /b 1
)

echo === Live docs at http://127.0.0.1:8000 ===
echo Press Ctrl+C to stop.
sphinx-autobuild docs\ docs\_build\ --watch src\qabit

endlocal
