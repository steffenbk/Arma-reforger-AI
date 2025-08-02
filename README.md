<img width="3814" height="1880" alt="image" src="https://github.com/user-attachments/assets/59a210a9-1ba0-4308-8d35-c0cb641351b7" />




# Arma-reforger-AI

This project is a modular, AI-powered Retrieval-Augmented Generation (RAG) system designed to assist Arma Reforger modding. It combines intelligent document search, chat-based interaction, and persistent memory to provide accurate, fast, and context-aware responses to modding-related queries.

Video:https://www.youtube.com/watch?v=srWNBHWiw6M



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
### Step 4: Organize Files
Create a base folder to store it, for example:
```
C:\ArmaModdingRAG
```
---

### Step 5: Install Files
- Download and extract the ZIP from Google Drive: https://drive.google.com/file/d/1jngx6k3VfFAKt5vJaSr4pBHnfgAvMcuT/view?usp=drive_link
---


### Step 6: Launch the Server

- Ensure Ollama is running
- Run `simple_launcher_server.bat` (You may get a Windows warning; this is normal)

Wait for the command window to complete loading (1–2 minutes depending on your hardware)

---

### Step 7: Launch the Interface

- Go into the **Webui** folder and find **index.html**
- Double-click on the HTML file to open the interface in your browser

---

## Notes

- Do **not** rename any of the directory structures inside the documentation folder



**example usecase**

<img width="3828" height="1819" alt="image" src="https://github.com/user-attachments/assets/77026735-9f24-4b76-9e3e-51875d7c90b7" />

