@echo off
echo ========================================
echo Food Health Risk Analyzer - Startup
echo ========================================

echo.
echo Starting FastAPI Backend...
start "FastAPI Backend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 5050"

echo.
echo Starting React Frontend...
timeout /t 3 /nobreak >nul
start "React Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo ========================================
echo Servers starting...
echo   - Backend: http://localhost:8001
echo   - Frontend: http://localhost:5173
echo   - API Docs: http://localhost:8001/docs
echo ========================================
echo.
pause