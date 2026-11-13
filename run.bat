@echo off
echo ========================================================
echo   Starting DeepGuard AI - Deepfake Detection Platform
echo   Web UI will be available at: http://127.0.0.1:8000
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
