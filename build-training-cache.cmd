@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0commands\build-training-cache.ps1" %*
pause
