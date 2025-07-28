@echo off
echo 🚀 Starting Arma RAG System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python and add to PATH
    pause
    exit /b 1
)

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

REM Start the API server in background
echo 🚀 Starting API server...
start "Arma RAG API" python main.py

REM Wait a bit for API to start
echo ⏳ Waiting for API to initialize...
timeout /t 8 /nobreak >nul

REM Change to WebUI directory and start Streamlit
echo 🌐 Starting WebUI...
cd "C:\ArmaModdingRAG\Python scripts\Amra rag delt opp\Webui"
streamlit run arma_rag_webui.py

echo.
echo 💡 Both services should now be running!
echo 🌐 API: http://localhost:8000
echo 🌐 WebUI: http://localhost:8501
echo.
pause