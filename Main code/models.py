from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    max_docs: Optional[int] = None  # No default - API handles this from config
    categories: Optional[List[str]] = None
    use_smart_processing: Optional[bool] = True

class ChatResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    context_used: int
    sources: List[Dict[str, str]]
    processing_time: float
    timestamp: str
    conversation_length: int

class EnhancedChatResponse(ChatResponse):
    query_type: str
    processed_query: str
    suggested_follow_ups: List[str]
    confidence_score: float
    multi_part: bool = False
    corrections_made: List[str] = []
    category_prefix_used: Optional[str] = None
    forced_categories: Optional[List[str]] = None
    intent_detected: Optional[str] = None
    context_enhanced: bool = False
    validation_issues: List[str] = []

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

class SystemStats(BaseModel):
    total_documents: int
    categories: Dict[str, Dict[str, Any]]
    vector_stores: List[str]
    system_status: str
    active_sessions: int
    total_conversations: int