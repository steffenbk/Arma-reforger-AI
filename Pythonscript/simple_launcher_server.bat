@echo off
echo 🚀 Starting Arma RAG API Server...
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
echo 💡 Please run auto_installer.bat first to install Python and dependencies
pause
exit /b 1

:python_found
echo ✅ Found Python: %PYTHON_CMD%

REM Check if Ollama is running
echo 🔍 Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not running! Please start Ollama first
    echo 💡 Run: ollama serve
    echo 💡 Then: ollama pull qwen3:14b
    pause
    exit /b 1
)

echo ✅ Ollama is running!
echo.

REM Start the API server
echo 🚀 Starting API server...
echo 🌐 API will be available at: http://localhost:8000
echo 📖 API docs at: http://localhost:8000/docs
echo.
echo ⚠️  Keep this window open while using the system
echo 💡 Press Ctrl+C to stop the API server
echo.

%PYTHON_CMD% main.py

pause