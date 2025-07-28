# Arma-reforger-AI

This project is a modular, AI-powered Retrieval-Augmented Generation (RAG) system designed to assist Arma Reforger modding. It combines intelligent document search, chat-based interaction, and persistent memory to provide accurate, fast, and context-aware responses to modding-related queries.


Documentation download must have: https://drive.google.com/file/d/1j4G_hvaeDo1HT3v3wYF-ywOWve3JDsxr/view?usp=sharing




Quick Install Steps
Step 1: Install Python

Go to python.org/downloads
Download Python 3.11 or newer
IMPORTANT: Check  "Add Python to PATH" during installation
Complete the installation

Step 2: Install Required Packages
Open Command Prompt and run this single command:
pip install fastapi uvicorn streamlit langchain langchain-chroma langchain-huggingface chromadb sentence-transformers torch transformers numpy pandas pydantic requests python-multipart


Step 3: Install Ollama (AI Engine)

Download Ollama from ollama.ai https://ollama.com/
Install Ollama
We will use the qwen3 model: https://ollama.com/library/qwen3
Open powershell or cmd type "ollama run qwen3:14b"


Step 4: Install files from the github, and the google drive zip.

Step 5:  Store all the files somewhere example in: C:\ArmaModdingRAG
Now extract all of the contents here.the folder you should have:
C:\ArmaModdingRAG\Arma_Reforger_RAG_Organized
C:\ArmaModdingRAG\chroma_db
the python scrip files you can just add into the C:\ArmaModdingRAG

Step 6: Right click on Config.py and edit with notepad or what you have.
Change these 3 to fit your documentation path:
    documents_path: str = r"C:\ArmaModdingRAG\Arma_Reforger_RAG_Organized"
    vector_db_path: str = r"C:\ArmaModdingRAG\chroma_db"
    memory_db_path: str = r"C:\ArmaModdingRAG\conversations.db" <--- this file will be created you dont need to make it

Step 7: Have ollama running, and launch the simple_launcher_server.bat you may get a windows warning but dont mind that
after that has started you will see a command window pop up, let it finish(takes 1 or 2 min depending on hardware)

Step 8: Double click on the html file to open it in your browser
    

