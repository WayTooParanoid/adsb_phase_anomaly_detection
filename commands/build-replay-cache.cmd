@echo off
echo Building dashboard-ready replay cache...
echo This uses the raw/processed ADS-B day folders and writes the fast runtime cache.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-replay-cache.ps1" %*
if errorlevel 1 (
  echo.
  echo Replay cache build failed. Review the error above.
  pause
  exit /b 1
)
echo.
echo Replay cache build finished.
pause
