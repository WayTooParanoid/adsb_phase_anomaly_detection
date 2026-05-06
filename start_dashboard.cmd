@echo off
setlocal
cd /d "%~dp0"

set "PORT=8050"
set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
  echo Python 3.11 or newer is required to run this dashboard.
  echo Install Python from https://www.python.org/downloads/ and run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (
    echo Failed to create the local Python environment.
    pause
    exit /b 1
  )
)

set "DASHBOARD_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist ".venv\.dashboard_deps_installed" (
  echo Installing dashboard dependencies. This first run can take a few minutes...
  "%DASHBOARD_PYTHON%" -m pip install --upgrade pip
  if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
  )
  "%DASHBOARD_PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Failed to install dependencies. Check your internet connection and Python installation.
    pause
    exit /b 1
  )
  echo ok> ".venv\.dashboard_deps_installed"
)

set "PYTHONPATH=%CD%\src"

echo Stopping any existing dashboard on port %PORT%...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\commands\stop-dashboard.ps1" -Port %PORT% >nul 2>nul

echo Starting dashboard on http://127.0.0.1:%PORT% ...
start "ADS-B Phase DBSCAN Dashboard" /D "%CD%" "%DASHBOARD_PYTHON%" -m adsb_phase_dbscan.live_dashboard --models-dir artifacts\models --config config.real.yaml --port %PORT% %*

echo Waiting for dashboard server...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0; $i -lt 45; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/' -TimeoutSec 2; if($r.StatusCode -ge 200){ $ok=$true; break } } catch { Start-Sleep -Seconds 1 } }; if($ok){ exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo.
  echo Dashboard did not become reachable at http://127.0.0.1:%PORT% .
  echo Check the "ADS-B Phase DBSCAN Dashboard" console window for the Python error.
  pause
  exit /b 1
)

start "" "http://127.0.0.1:%PORT%"
