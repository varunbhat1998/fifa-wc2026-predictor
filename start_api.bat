@echo off
REM Persistent API server. Auto-restarts if it crashes. Close the window to stop.
cd /d "%~dp0"
title FIFA Predictor API (port 8001)
:loop
echo [%date% %time%] Starting API server...
python 06_api.py
echo [%date% %time%] API exited. Restarting in 5 seconds. Press Ctrl+C to quit.
timeout /t 5
goto loop
