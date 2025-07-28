import os
import sqlite3
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import APIConfig

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation memory with optional persistence"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.sessions = {}  # In-memory session storage
        self.db_path = config.memory_db_path
        
        if config.enable_persistent_memory:
            self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent memory"""
        try:
            # Ensure directory exists with better error handling
            db_dir = os.path.dirname(self.db_path)
            print(f"🗃️ Creating database directory: {db_dir}")
            
            if not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"✅ Created directory: {db_dir}")
            else:
                print(f"✅ Directory already exists: {db_dir}")
            
            # Test database connection
            print(f"🗃️ Initializing database: {self.db_path}")
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON conversations(session_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON conversations(session_id, timestamp)
                """)
                
                # Test write/read
                conn.execute("INSERT OR IGNORE INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                           ("test", "system", "Database initialization test", datetime.now().isoformat()))
                
                # Check if we can read back
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                count = cursor.fetchone()[0]
                
                print(f"✅ Database initialized successfully with {count} total messages")
                
            logger.info("✅ Conversation database initialized")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            print(f"❌ Database path: {self.db_path}")
            print(f"❌ Directory path: {os.path.dirname(self.db_path)}")
            logger.error(f"❌ Failed to initialize conversation database: {e}")
            # Disable persistent memory if database fails
            self.config.enable_persistent_memory = False
            print("⚠️ Disabling persistent memory due to database error")
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            # Load from database if persistent memory is enabled
            if self.config.enable_persistent_memory:
                self.sessions[session_id] = self._load_session_from_db(session_id)
            else:
                self.sessions[session_id] = []
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation"""
        timestamp = datetime.now().isoformat()
        
        # Add to in-memory storage
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        
        # Maintain conversation length limit
        if len(self.sessions[session_id]) > self.config.max_conversation_length * 2:  # *2 because user+assistant
            self.sessions[session_id] = self.sessions[session_id][-self.config.max_conversation_length * 2:]
        
        # Save to database if persistent memory is enabled
        if self.config.enable_persistent_memory:
            self._save_message_to_db(session_id, role, content, timestamp)
    
    def get_conversation_context(self, session_id: str, max_exchanges: int = 5) -> str:
        """Get formatted conversation context for prompt - FIXED to prevent hallucination"""
        if session_id not in self.sessions or not self.sessions[session_id]:
            return "No conversation history available."
        
        messages = self.sessions[session_id]
        
        # Get last N exchanges (user + assistant pairs)
        recent_messages = messages[-(max_exchanges * 2):]
        
        if not recent_messages:
            return "No conversation history available."
        
        # Format more clearly to prevent hallucination
        context_parts = ["=== ACTUAL CONVERSATION HISTORY ==="]
        
        for i, msg in enumerate(recent_messages, 1):
            timestamp = msg['timestamp'][:19].replace('T', ' ')
            # Clean the content to prevent LLM confusion
            clean_content = msg['content'].replace('<think>', '').replace('</think>', '').strip()
            if clean_content:  # Only add non-empty messages
                context_parts.append(f"Message {i} [{timestamp}] - {msg['role'].upper()}: {clean_content}")
        
        context_parts.append("=== END OF ACTUAL CONVERSATION ===")
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for the session"""
        if session_id not in self.sessions:
            return {
                "length": 0,
                "last_activity": None,
                "topics": []
            }
        
        messages = self.sessions[session_id]
        
        return {
            "length": len(messages) // 2,  # Number of exchanges
            "last_activity": messages[-1]["timestamp"] if messages else None,
            "topics": self._extract_topics(messages)
        }
    
    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        """Extract main topics from conversation (simple keyword extraction)"""
        topics = set()
        # Arma Reforger specific keywords
        keywords = [
            "weapon", "ai", "vehicle", "component", "script", "class", "function", 
            "tutorial", "guide", "arma", "reforger", "enfusion", "scr_", "modding",
            "workbench", "editor", "damage", "health", "physics", "animation",
            "bones", "skeleton", "import", "export", "blender", "model", "texture"
        ]
        
        for msg in messages:
            if msg["role"] == "user":
                content_lower = msg["content"].lower()
                for keyword in keywords:
                    if keyword in content_lower:
                        topics.add(keyword)
        
        return list(topics)[:5]  # Return top 5 topics
    
    def _load_session_from_db(self, session_id: str) -> List[Dict]:
        """Load session from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT role, content, timestamp FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, self.config.max_conversation_length * 2))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        "role": row[0],
                        "content": row[1], 
                        "timestamp": row[2]
                    })
                
                return list(reversed(messages))  # Return in chronological order
                
        except Exception as e:
            logger.error(f"❌ Error loading session {session_id}: {e}")
            return []
    
    def _save_message_to_db(self, session_id: str, role: str, content: str, timestamp: str):
        """Save message to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO conversations (session_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (session_id, role, content, timestamp))
                
        except Exception as e:
            logger.error(f"❌ Error saving message to database: {e}")
    
    def get_active_sessions_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)
    
    def get_total_conversations_count(self) -> int:
        """Get total number of conversations from database"""
        if not self.config.enable_persistent_memory:
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
                return cursor.fetchone()[0]
        except:
            return 0
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_raw_conversation_history(self, session_id: str) -> List[Dict]:
        """Get raw conversation messages for debugging"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id].copy()
    
    def debug_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get debug information about a session"""
        if session_id not in self.sessions:
            return {"error": "Session not found", "session_id": session_id}
        
        messages = self.sessions[session_id]
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "first_message": messages[0] if messages else None,
            "last_message": messages[-1] if messages else None,
            "topics_detected": self._extract_topics(messages),
            "raw_messages": messages
        }