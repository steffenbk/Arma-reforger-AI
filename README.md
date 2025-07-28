# Arma-reforger-AI

This project is a modular, AI-powered Retrieval-Augmented Generation (RAG) system designed to assist Arma Reforger modding. It combines intelligent document search, chat-based interaction, and persistent memory to provide accurate, fast, and context-aware responses to modding-related queries.


Documentation download must have: https://drive.google.com/file/d/1j4G_hvaeDo1HT3v3wYF-ywOWve3JDsxr/view?usp=sharing




# Arma Reforger AI

This project is a modular, AI-powered Retrieval-Augmented Generation (RAG) system designed to assist Arma Reforger modding. It combines intelligent document search, chat-based interaction, and persistent memory to provide accurate, fast, and context-aware responses to modding-related queries.

**Documentation Download (Required):**  
[Download from Google Drive](https://drive.google.com/file/d/1j4G_hvaeDo1HT3v3wYF-ywOWve3JDsxr/view?usp=sharing)

---

## Quick Install Steps

### Step 1: Install Python
- Go to [python.org/downloads](https://www.python.org/downloads)
- Download Python **3.11 or newer**
- **IMPORTANT:** Check **"Add Python to PATH"** during installation
- Complete the installation

---

### Step 2: Install Required Packages
Open Command Prompt and run this single command:

```bash
pip install fastapi uvicorn streamlit langchain langchain-chroma langchain-huggingface chromadb sentence-transformers torch transformers numpy pandas pydantic requests python-multipart
```

---

### Step 3: Install Ollama (AI Engine)
- Download from: [https://ollama.com](https://ollama.com/)
- Install Ollama
- We will use the **qwen3 model**: [https://ollama.com/library/qwen3](https://ollama.com/library/qwen3)

Run this in PowerShell or CMD:
```bash
ollama run qwen3:14b
```

---

### Step 4: Install Files
- Clone or download this GitHub repository
- Download and extract the documentation ZIP from Google Drive

---

### Step 5: Organize Files
Create a base directory, for example:
```
C:\ArmaModdingRAG
```

Inside that directory, place the following:
```
C:\ArmaModdingRAG\Arma_Reforger_RAG_Organized
C:\ArmaModdingRAG\chroma_db
```

Place the Python script files (from GitHub) directly into:
```
C:\ArmaModdingRAG
```

---

### Step 6: Configure `Config.py`

Open `Config.py` with a text editor and update the following paths:

```python
documents_path: str = r"C:\ArmaModdingRAG\Arma_Reforger_RAG_Organized"
vector_db_path: str = r"C:\ArmaModdingRAG\chroma_db"
memory_db_path: str = r"C:\ArmaModdingRAG\conversations.db"  # (This file will be created automatically)
```

---

### Step 7: Launch the Server

- Ensure Ollama is running
- Run `simple_launcher_server.bat` (You may get a Windows warning; this is normal)

Wait for the command window to complete loading (1–2 minutes depending on your hardware)

---

### Step 8: Launch the Interface

- Double-click on the HTML file to open the interface in your browser

---

## Notes

- Do **not** rename any of the directory structures inside the documentation folder
- If you update the documentation, you may need to delete and regenerate the `chroma_db` folder
