@echo off
REM Build the Sphinx documentation and serve it at http://localhost:8000
REM Stop the server with Ctrl+C.

setlocal
cd /d "%~dp0\.."

echo === Building docs ===
sphinx-build docs\ docs\_build\
if errorlevel 1 (
    echo.
    echo Docs build failed. If sphinx is missing, install the docs extras:
    echo     pip install -e ".[docs]"
    exit /b 1
)

echo.
echo === Serving docs at http://localhost:8000 ===
echo Press Ctrl+C to stop.
where py >nul 2>&1
if %errorlevel%==0 (
    py -m http.server -d docs\_build 8000
) else (
    python -m http.server -d docs\_build 8000
)

endlocal
