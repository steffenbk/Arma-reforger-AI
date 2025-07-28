import logging
import threading
import time
import uvicorn

from config import APIConfig
from api_routes import ArmaRAGAPI  # ← Changed from 'api' to 'api_routes'
from console import ConsoleChatInterface

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_api_server():
    """Start the API server"""
    config = APIConfig()
    api = ArmaRAGAPI(config)
    
    print(f"🚀 Starting Arma RAG API on {config.api_host}:{config.api_port}")
    
    uvicorn.run(
        api.app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )

def start_console_chat():
    """Start the console chat interface (synchronous version)"""
    chat = ConsoleChatInterface()
    chat.start_chat()

def start_api_server_background():
    """Start API server in background"""
    def run_server():
        config = APIConfig()
        api = ArmaRAGAPI(config)
        uvicorn.run(
            api.app,
            host=config.api_host,
            port=config.api_port,
            log_level="error"  # Reduce noise
        )
    
    # Start API server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("🚀 Starting Arma RAG API server...")
    time.sleep(5)  # Give server more time to start
    
    print("✅ API server started in background")
    print("🎯 API available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")

def main():
    """Start both API server and console chat"""
    print("🚀 Arma Reforger RAG System - API & Console Chat")
    print("=" * 60)
    
    try:
        # Start API server in background
        start_api_server_background()
        
        # Start console chat
        print("\n💬 Starting console chat interface...")
        start_console_chat()
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()