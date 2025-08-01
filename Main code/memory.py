import os
import sqlite3
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from config import APIConfig

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a complete user question + AI response pair"""
    user_message: str
    ai_response: str
    timestamp: str
    topics: List[str]
    question_type: str  # "initial", "follow_up", "clarification", etc.
    reference_context: Optional[str] = None  # What this turn references

class EnhancedConversationMemory:
    """Enhanced conversation memory with better context understanding"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.sessions = {}  # In-memory session storage
        self.conversation_turns = {}  # Structured Q&A pairs
        self.db_path = config.memory_db_path
        
        if config.enable_persistent_memory:
            self._init_database()
    
    def _init_database(self):
        """Initialize enhanced SQLite database"""
        try:
            db_dir = os.path.dirname(self.db_path)
            
            # Handle case where database is in current directory
            if not db_dir:
                db_dir = "."
                print(f"🗃️ Database will be created in current directory: {os.path.abspath(self.db_path)}")
            else:
                print(f"🗃️ Creating database directory: {db_dir}")
                
                if not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                    print(f"✅ Created directory: {db_dir}")
                else:
                    print(f"✅ Directory already exists: {db_dir}")
            
            # Test database connection
            print(f"🗃️ Initializing database: {self.db_path}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Original messages table
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
                
                # NEW: Conversation turns table for better structure
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn_number INTEGER NOT NULL,
                        user_message TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        topics TEXT,  -- JSON array of topics
                        question_type TEXT,
                        reference_context TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id)")
                
            logger.info("✅ Enhanced conversation database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced database: {e}")
            self.config.enable_persistent_memory = False
    
    def add_conversation_turn(self, session_id: str, user_message: str, ai_response: str):
        """Add a complete conversation turn (user + AI response)"""
        timestamp = datetime.now().isoformat()
        
        # Analyze the question type and context
        question_type = self._classify_question_type(user_message, session_id)
        topics = self._extract_enhanced_topics(user_message, ai_response)
        reference_context = self._detect_reference_context(user_message, session_id)
        
        # Create conversation turn
        turn = ConversationTurn(
            user_message=user_message,
            ai_response=ai_response,
            timestamp=timestamp,
            topics=topics,
            question_type=question_type,
            reference_context=reference_context
        )
        
        # Store in memory
        if session_id not in self.conversation_turns:
            self.conversation_turns[session_id] = []
        
        self.conversation_turns[session_id].append(turn)
        
        # Maintain turn limit (keep last N complete turns)
        max_turns = self.config.max_conversation_length
        if len(self.conversation_turns[session_id]) > max_turns:
            self.conversation_turns[session_id] = self.conversation_turns[session_id][-max_turns:]
        
        # Save to database
        if self.config.enable_persistent_memory:
            self._save_turn_to_db(session_id, turn)
        
        # Also maintain the original message format for compatibility
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", ai_response)
    
    def _classify_question_type(self, user_message: str, session_id: str) -> str:
        """Classify the type of question to understand context needs"""
        message_lower = user_message.lower().strip()
        
        # Check if it's referencing previous conversation
        reference_indicators = [
            "that", "this", "it", "the last", "previous", "earlier", "before",
            "you said", "you mentioned", "from above", "the answer", "your response",
            "build on", "expand", "explain more", "tell me more", "elaborate",
            "can you", "could you", "what about", "how about", "also"
        ]
        
        follow_up_patterns = [
            "can you explain", "tell me more", "elaborate", "expand on",
            "build on", "what about", "how about", "also", "and what",
            "but what", "why", "how does", "what if", "can i also"
        ]
        
        clarification_patterns = [
            "what do you mean", "i don't understand", "unclear", "confusing",
            "can you clarify", "what is", "which", "where exactly"
        ]
        
        # Check for follow-up questions
        if any(pattern in message_lower for pattern in follow_up_patterns):
            return "follow_up"
        
        # Check for clarification requests
        if any(pattern in message_lower for pattern in clarification_patterns):
            return "clarification"
        
        # Check for reference to previous context
        if any(indicator in message_lower for indicator in reference_indicators):
            return "contextual_reference"
        
        # Check if this is the first question in session
        if session_id not in self.conversation_turns or not self.conversation_turns[session_id]:
            return "initial"
        
        return "new_topic"
    
    def _detect_reference_context(self, user_message: str, session_id: str) -> Optional[str]:
        """Detect what the user is referring to from previous conversation"""
        if session_id not in self.conversation_turns or not self.conversation_turns[session_id]:
            return None
        
        message_lower = user_message.lower()
        recent_turns = self.conversation_turns[session_id][-3:]  # Last 3 turns
        
        # Look for specific references
        reference_patterns = {
            "last answer": "previous_response",
            "your response": "previous_response", 
            "you said": "previous_response",
            "that solution": "previous_solution",
            "this method": "previous_method",
            "the code": "previous_code",
            "my question": "previous_question",
            "the problem": "previous_problem"
        }
        
        for pattern, ref_type in reference_patterns.items():
            if pattern in message_lower:
                return f"{ref_type}:{recent_turns[-1].ai_response[:200]}..." if recent_turns else None
        
        return None
    
    def _extract_enhanced_topics(self, user_message: str, ai_response: str) -> List[str]:
        """Enhanced topic extraction from both question and answer"""
        topics = set()
        
        # Technical Arma Reforger terms
        technical_terms = [
            "weapon", "rifle", "pistol", "ai", "vehicle", "component", "script", "class", 
            "function", "arma", "reforger", "enfusion", "scr_", "modding", "workbench", 
            "editor", "damage", "health", "physics", "animation", "bones", "skeleton", 
            "import", "export", "blender", "model", "texture", "material", "shader",
            "entity", "prefab", "world", "hierarchy", "inheritance", "override"
        ]
        
        # Problem/solution indicators
        problem_indicators = [
            "error", "issue", "problem", "not working", "broken", "missing", "failed",
            "can't", "won't", "doesn't work", "stuck", "crash", "bug"
        ]
        
        # Solution indicators  
        solution_indicators = [
            "fix", "solve", "solution", "enable", "disable", "configure", "setup",
            "implement", "create", "add", "remove", "modify", "update"
        ]
        
        combined_text = f"{user_message} {ai_response}".lower()
        
        # Extract technical terms
        for term in technical_terms:
            if term in combined_text:
                topics.add(term)
        
        # Detect problem/solution nature
        if any(indicator in combined_text for indicator in problem_indicators):
            topics.add("troubleshooting")
        
        if any(indicator in combined_text for indicator in solution_indicators):
            topics.add("implementation")
        
        return list(topics)[:8]  # Top 8 topics
    
    def get_enhanced_conversation_context(self, session_id: str, user_message: str) -> str:
        """Get intelligently formatted context based on question type"""
        if session_id not in self.conversation_turns or not self.conversation_turns[session_id]:
            return "No previous conversation history."
        
        question_type = self._classify_question_type(user_message, session_id)
        turns = self.conversation_turns[session_id]
        
        # Build context based on question type
        if question_type in ["follow_up", "clarification", "contextual_reference"]:
            # For follow-ups, emphasize the most recent exchange
            context_parts = ["=== CONVERSATION CONTEXT FOR FOLLOW-UP QUESTION ==="]
            
            # Always include the most recent complete exchange
            if turns:
                last_turn = turns[-1]
                context_parts.extend([
                    "",
                    "📝 MOST RECENT EXCHANGE:",
                    f"👤 USER ASKED: {last_turn.user_message}",
                    f"🤖 AI RESPONDED: {last_turn.ai_response[:500]}{'...' if len(last_turn.ai_response) > 500 else ''}",
                    ""
                ])
                
                # Add topics for better context
                if last_turn.topics:
                    context_parts.append(f"🏷️ TOPICS DISCUSSED: {', '.join(last_turn.topics)}")
                
                # If there are earlier relevant turns, add them
                if len(turns) > 1:
                    context_parts.extend([
                        "",
                        "📚 EARLIER CONTEXT:"
                    ])
                    
                    for turn in turns[-3:-1]:  # Previous 2 turns before the last one
                        context_parts.append(f"• {turn.user_message[:100]}... → {turn.ai_response[:100]}...")
            
            context_parts.extend([
                "",
                f"❓ CURRENT FOLLOW-UP: {user_message}",
                "=== END CONTEXT ==="
            ])
            
        else:
            # For new topics, provide general conversation summary
            context_parts = ["=== GENERAL CONVERSATION CONTEXT ==="]
            
            if turns:
                # Summarize recent topics
                all_topics = set()
                for turn in turns[-3:]:
                    all_topics.update(turn.topics)
                
                if all_topics:
                    context_parts.append(f"🏷️ RECENT TOPICS: {', '.join(list(all_topics)[:10])}")
                
                # Show last exchange briefly
                last_turn = turns[-1]
                context_parts.extend([
                    "",
                    f"📝 LAST EXCHANGE: {last_turn.user_message[:150]}... → Solution provided",
                    ""
                ])
            
            context_parts.extend([
                f"❓ NEW QUESTION: {user_message}",
                "=== END CONTEXT ==="
            ])
        
        return "\n".join(context_parts)
    
    def analyze_conversation_for_memory_query(self, session_id: str, memory_query: str) -> str:
        """Analyze conversation history to answer memory queries"""
        if session_id not in self.conversation_turns or not self.conversation_turns[session_id]:
            return "No conversation history found for this session."
        
        turns = self.conversation_turns[session_id]
        query_lower = memory_query.lower()
        
        # Build comprehensive conversation analysis
        analysis_parts = [
            "🧠 CONVERSATION MEMORY ANALYSIS",
            "=" * 40,
            ""
        ]
        
        # Summary stats
        analysis_parts.extend([
            f"📊 CONVERSATION OVERVIEW:",
            f"• Total exchanges: {len(turns)}",
            f"• Duration: {self._calculate_conversation_duration(turns)}",
            f"• Main topics: {', '.join(self._get_dominant_topics(turns))}",
            ""
        ])
        
        # Recent context
        if "recent" in query_lower or "last" in query_lower:
            last_turn = turns[-1]
            analysis_parts.extend([
                "📝 MOST RECENT EXCHANGE:",
                f"👤 Question: {last_turn.user_message}",
                f"🤖 Response: {last_turn.ai_response[:300]}{'...' if len(last_turn.ai_response) > 300 else ''}",
                f"🏷️ Topics: {', '.join(last_turn.topics)}",
                ""
            ])
        
        # Topic-based search
        if any(topic in query_lower for topic in ["weapon", "ai", "vehicle", "script", "error", "problem"]):
            relevant_turns = [turn for turn in turns if any(topic in query_lower for topic in turn.topics)]
            if relevant_turns:
                analysis_parts.extend([
                    f"🔍 RELEVANT DISCUSSIONS:",
                    ""
                ])
                for turn in relevant_turns[-3:]:  # Last 3 relevant
                    analysis_parts.append(f"• {turn.user_message[:100]}... → {turn.question_type}")
                analysis_parts.append("")
        
        # Conversation flow analysis
        analysis_parts.extend([
            "🔄 CONVERSATION FLOW:",
            ""
        ])
        
        for i, turn in enumerate(turns[-5:], 1):  # Last 5 turns
            question_type_emoji = {
                "initial": "🆕",
                "follow_up": "➡️", 
                "clarification": "❓",
                "contextual_reference": "🔗",
                "new_topic": "📋"
            }
            
            emoji = question_type_emoji.get(turn.question_type, "💬")
            analysis_parts.append(f"{emoji} Turn {i}: {turn.user_message[:80]}...")
        
        return "\n".join(analysis_parts)
    
    def _calculate_conversation_duration(self, turns: List[ConversationTurn]) -> str:
        """Calculate how long the conversation has been going"""
        if not turns:
            return "No messages"
        
        try:
            first_time = datetime.fromisoformat(turns[0].timestamp)
            last_time = datetime.fromisoformat(turns[-1].timestamp)
            duration = last_time - first_time
            
            if duration.seconds < 60:
                return f"{duration.seconds} seconds"
            elif duration.seconds < 3600:
                return f"{duration.seconds // 60} minutes"
            else:
                return f"{duration.seconds // 3600} hours, {(duration.seconds % 3600) // 60} minutes"
        except:
            return "Unknown duration"
    
    def _get_dominant_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Get the most frequently mentioned topics"""
        topic_counts = {}
        for turn in turns:
            for topic in turn.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by frequency and return top 5
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def get_contextual_prompt_enhancement(self, session_id: str, user_message: str) -> str:
        """Generate enhanced prompt context for better AI understanding"""
        if session_id not in self.conversation_turns or not self.conversation_turns[session_id]:
            return ""
        
        question_type = self._classify_question_type(user_message, session_id)
        turns = self.conversation_turns[session_id]
        
        if question_type in ["follow_up", "clarification", "contextual_reference"]:
            last_turn = turns[-1]
            
            return f"""
IMPORTANT CONTEXT: This is a {question_type.replace('_', ' ')} question referring to our previous discussion.

PREVIOUS DISCUSSION:
User asked: "{last_turn.user_message}"
AI provided: "{last_turn.ai_response[:400]}{'...' if len(last_turn.ai_response) > 400 else ''}"
Topics covered: {', '.join(last_turn.topics)}

CURRENT REQUEST: "{user_message}"

Please build upon the previous discussion and provide a response that acknowledges and extends the previous context.
"""
        
        return ""
    
    # Keep all the original methods for compatibility
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            if self.config.enable_persistent_memory:
                self.sessions[session_id] = self._load_session_from_db(session_id)
            else:
                self.sessions[session_id] = []
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation (original compatibility method)"""
        timestamp = datetime.now().isoformat()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        
        # Maintain conversation length limit
        if len(self.sessions[session_id]) > self.config.max_conversation_length * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.config.max_conversation_length * 2:]
        
        if self.config.enable_persistent_memory:
            self._save_message_to_db(session_id, role, content, timestamp)
    
    def get_conversation_context(self, session_id: str, max_exchanges: int = 5) -> str:
        """Enhanced conversation context that handles follow-ups better"""
        # Use the new enhanced context method
        return self.get_enhanced_conversation_context(session_id, "")
    
    def _save_turn_to_db(self, session_id: str, turn: ConversationTurn):
        """Save conversation turn to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                turn_number = len(self.conversation_turns[session_id])
                
                conn.execute("""
                    INSERT INTO conversation_turns 
                    (session_id, turn_number, user_message, ai_response, timestamp, topics, question_type, reference_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, turn_number, turn.user_message, turn.ai_response,
                    turn.timestamp, json.dumps(turn.topics), turn.question_type, turn.reference_context
                ))
                
        except Exception as e:
            logger.error(f"❌ Error saving conversation turn: {e}")
    
    # Keep other original methods for compatibility...
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
                
                return list(reversed(messages))
                
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
    
    # Other utility methods remain the same...
    def get_active_sessions_count(self) -> int:
        return len(self.sessions)
    
    def get_total_conversations_count(self) -> int:
        if not self.config.enable_persistent_memory:
            return 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
                return cursor.fetchone()[0]
        except:
            return 0
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for the session"""
        if session_id not in self.conversation_turns:
            return {
                "length": 0,
                "last_activity": None,
                "topics": []
            }
        
        turns = self.conversation_turns[session_id]
        
        return {
            "length": len(turns),  # Number of exchanges
            "last_activity": turns[-1].timestamp if turns else None,
            "topics": self._get_dominant_topics(turns)
        }
    
    def get_raw_conversation_history(self, session_id: str) -> List[Dict]:
        """Get raw conversation messages for debugging"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id].copy()
    
    def debug_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get debug information about a session"""
        if session_id not in self.conversation_turns:
            return {"error": "Session not found", "session_id": session_id}
        
        turns = self.conversation_turns[session_id]
        messages = self.sessions.get(session_id, [])
        
        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "message_count": len(messages),
            "first_turn": turns[0] if turns else None,
            "last_turn": turns[-1] if turns else None,
            "topics_detected": self._get_dominant_topics(turns),
            "raw_messages": messages
        }
    
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
        if session_id in self.conversation_turns:
            del self.conversation_turns[session_id]