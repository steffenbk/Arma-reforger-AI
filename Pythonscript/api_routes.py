import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import APIConfig
from models import ChatMessage, EnhancedChatResponse, SystemStats
from api_processor import APIProcessor

logger = logging.getLogger(__name__)

class ArmaRAGAPI:
    """FastAPI application with route definitions - delegates processing to APIProcessor"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.processor = APIProcessor(config)  # Delegate all processing
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Arma Reforger RAG API",
            description="AI-powered Arma Reforger modding assistant with conversation memory and smart query processing",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    @property
    def system_ready(self) -> bool:
        """Check if system is ready via processor"""
        return self.processor.system_ready
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Arma Reforger RAG API",
                "status": "ready" if self.system_ready else "initializing",
                "endpoints": {
                    "chat": "/chat",
                    "stats": "/stats",
                    "health": "/health",
                    "conversation": "/conversation/{session_id}",
                    "clear_session": "/conversation/{session_id}/clear"
                }
            }
        
        @self.app.post("/chat", response_model=EnhancedChatResponse)
        async def chat(message: ChatMessage):
            """Main chat endpoint - delegates to processor"""
            if not self.system_ready:
                raise HTTPException(status_code=503, detail="System is still initializing")
            
            try:
                logger.info(f"🔧 Processing: '{message.message[:50]}...'")
                return await self.processor.process_chat_message(message)
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/conversation/{session_id}")
        async def get_conversation(session_id: str):
            """Get conversation history for a session"""
            if not self.system_ready:
                raise HTTPException(status_code=503, detail="System is still initializing")
            
            return await self.processor.get_conversation(session_id)
        
        @self.app.delete("/conversation/{session_id}/clear")
        async def clear_conversation(session_id: str):
            """Clear conversation history for a session"""
            if not self.system_ready:
                raise HTTPException(status_code=503, detail="System is still initializing")
            
            await self.processor.clear_conversation(session_id)
            return {"message": f"Conversation {session_id} cleared"}
        
        @self.app.get("/stats", response_model=SystemStats)
        async def get_stats():
            """Get system statistics"""
            if not self.system_ready:
                raise HTTPException(status_code=503, detail="System is still initializing")
            
            return await self.processor.get_system_stats()
        
        @self.app.get("/health")
        async def health_check():
            """System health check"""
            return await self.processor.health_check()