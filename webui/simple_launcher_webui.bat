@echo off
echo 🌐 Starting Arma RAG WebUI...
echo.

REM Find Python command
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)

echo ❌ Python not found! 
echo 💡 Please install Python first
pause
exit /b 1

:python_found
echo ✅ Found Python: %PYTHON_CMD%

REM Check if API is running
echo 🔍 Checking if API server is running...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API server not running!
    echo.
    echo 💡 Please start the API server first by running:
    echo    start_api_server.bat
    echo.
    echo 💡 Or make sure main.py is running on port 8000
    pause
    exit /b 1
)

echo ✅ API server is running!
echo.

REM Start Streamlit WebUI
echo 🚀 Starting WebUI...
echo 🌐 WebUI will be available at: http://localhost:8501
echo.
echo ⚠️  Keep this window open while using the WebUI
echo 💡 Press Ctrl+C to stop the WebUI
echo.

%PYTHON_CMD% -m streamlit run arma_rag_webui.py --server.address localhost --server.port 8501

pause