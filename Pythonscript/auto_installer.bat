@echo off
echo 🚀 Arma RAG System - Auto Installer
echo =====================================
echo.

REM Check if we're running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  This installer needs administrator privileges to install Python
    echo 🔄 Restarting as administrator...
    echo.
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ✅ Running as administrator
echo.

REM Step 1: Check/Install Python
echo 🐍 Step 1: Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Python not found - installing automatically...
        echo 📥 Downloading Python 3.11...
        
        REM Download and install Python silently
        curl -o "%TEMP%\python-installer.exe" "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
        if errorlevel 1 (
            echo ❌ Download failed! Please install Python manually from python.org
            pause
            exit /b 1
        )
        
        echo ⚙️ Installing Python (this may take a few minutes)...
        "%TEMP%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        
        echo 🗑️ Cleaning up...
        del "%TEMP%\python-installer.exe"
        
        REM Refresh PATH environment variable
        call refreshenv >nul 2>&1
        
        echo ✅ Python installed!
    ) else (
        echo ✅ Python found (py command)
        set PYTHON_CMD=py
    )
) else (
    echo ✅ Python found (python command)
    set PYTHON_CMD=python
)

echo.

REM Step 2: Install Python packages
echo 📦 Step 2: Installing required Python packages...
echo This may take several minutes on first run...
echo.

REM Try different Python commands
if defined PYTHON_CMD (
    %PYTHON_CMD% -m pip install --upgrade pip
    %PYTHON_CMD% -m pip install fastapi uvicorn streamlit langchain langchain-chroma langchain-huggingface chromadb sentence-transformers torch transformers numpy pandas pydantic requests python-multipart
) else (
    python -m pip install --upgrade pip
    python -m pip install fastapi uvicorn streamlit langchain langchain-chroma langchain-huggingface chromadb sentence-transformers torch transformers numpy pandas pydantic requests python-multipart
)

if errorlevel 1 (
    echo ❌ Package installation failed!
    echo 💡 Try running this manually: pip install fastapi uvicorn streamlit langchain
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!
echo.

REM Step 3: Check Ollama
echo 🤖 Step 3: Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not running or not installed
    echo.
    echo 📋 MANUAL STEPS REQUIRED:
    echo 1. Download Ollama from: https://ollama.ai/
    echo 2. Install Ollama
    echo 3. Open command prompt and run: ollama serve
    echo 4. In another command prompt run: ollama pull qwen3:14b
    echo 5. Then run this installer again
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Ollama is running!
)

REM Step 4: Create data directories
echo 📁 Step 4: Creating data directories...
if not exist "ArmaRAG_Data" mkdir "ArmaRAG_Data"
if not exist "ArmaRAG_Data\Arma_Reforger_RAG_Organized" mkdir "ArmaRAG_Data\Arma_Reforger_RAG_Organized"
if not exist "ArmaRAG_Data\chroma_db" mkdir "ArmaRAG_Data\chroma_db"
echo ✅ Data directories created!

echo.
echo 🎉 SETUP COMPLETE!
echo ================
echo.
echo 📋 Next steps:
echo 1. Copy your Arma Reforger documents to: ArmaRAG_Data\Arma_Reforger_RAG_Organized\
echo 2. Run start_arma_rag.bat to launch the system
echo.
echo 💡 The system is now ready to use!
echo.
pause