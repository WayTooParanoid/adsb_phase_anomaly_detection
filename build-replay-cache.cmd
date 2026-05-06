@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0commands\build-replay-cache.ps1" %*
pause
