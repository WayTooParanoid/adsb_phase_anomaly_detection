@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0commands\train-models.ps1" %*
pause
