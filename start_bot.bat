@echo off
REM Persistent Telegram bot. Auto-restarts if it crashes. Close the window to stop.
REM ⚠ The API server (start_api.bat) MUST be running first.
cd /d "%~dp0"
title FIFA Predictor Telegram Bot
:loop
echo [%date% %time%] Starting bot...
python -u match_bot.py
echo [%date% %time%] Bot exited. Restarting in 5 seconds. Press Ctrl+C to quit.
timeout /t 5
goto loop
