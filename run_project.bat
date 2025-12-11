@echo off
REM
cd /d "%~dp0"

echo Activating virtual environment...
call .\venv\Scripts\activate

echo Starting backend API server...
start cmd /k "cd /d %~dp0 && uvicorn backend:app --reload"

echo Starting static HTTP server on port 5500...
start cmd /k "cd /d %~dp0 && python -m http.server 5500"

echo Waiting for servers to start...
timeout /t 3 >nul

echo Running WiFi scan once...
python scan_current_wifi.py

echo Opening dashboard in browser...
start "" "http://127.0.0.1:5500/index.html?v=3"

echo All services started. Switch to browser and click 'Load last scan'.
