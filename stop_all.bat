@echo off
REM Kills every running python.exe — use when you want a clean restart.
taskkill /F /IM python.exe 2>nul
echo Stopped all python processes.
timeout /t 2
