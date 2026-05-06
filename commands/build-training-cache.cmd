@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-training-cache.ps1" %*
pause
