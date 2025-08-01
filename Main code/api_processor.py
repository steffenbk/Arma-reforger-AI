import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain.schema import Document

from config import APIConfig
from models import ChatMessage, EnhancedChatResponse, SystemStats
from memory import EnhancedConversationMemory  # CHANGED: Updated import
from smart_processing import QueryClassifier, QueryProcessor, MultiPartHandler, SuggestionEngine
from api_llm import LLMManager

logger = logging.getLogger(__name__)

class APIProcessor:
    """Handles all query processing, document retrieval, and business logic"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.memory = EnhancedConversationMemory(config)  # CHANGED: Updated memory class
        
        # Initialize smart processing components
        self.query_classifier = QueryClassifier()
        self.query_processor = QueryProcessor()
        self.multi_part_handler = MultiPartHandler()
        self.suggestion_engine = SuggestionEngine()
        
        # Initialize LLM manager (handles Ollama, system init, vector stores)
        self.llm_manager = LLMManager(config)
        
    @property
    def system_ready(self) -> bool:
        """Check if system is ready"""
        return self.llm_manager.system_ready
    
    @property
    def vector_stores(self) -> Dict:
        """Access vector stores via LLM manager"""
        return self.llm_manager.vector_stores
    
    def should_auto_search(self, query: str, conversation_context: List) -> tuple[bool, str]:
        """Detect if a conversational query actually needs document search with transparency"""
        
        query_lower = query.lower()
        
        # Edge case handling - explicit conversation references should stay in memory
        memory_indicators = [
            "our conversation", "what we discussed", "you just said", "you mentioned",
            "earlier you", "what you told me", "in our chat", "we talked about",
            "from our discussion", "you explained", "your previous answer"
        ]
        if any(indicator in query_lower for indicator in memory_indicators):
            return False, "conversation reference detected"
        
        # Context awareness - check if recent response was technical
        recent_technical = self._was_recent_response_technical(conversation_context)
        
        # Technical keywords that indicate need for documentation
        technical_keywords = [
            # Arma-specific
            "enfusion", "scr_", "workbench", "skeleton", "armature", "vertex groups",
            "fbx", "import", "export", "bones", "sockets", "hierarchy",
            "reforger", "arma", "modding", "mod", "asset", "resource",
            
            # General technical
            "component", "class", "function", "script", "code", "error",
            "settings", "configuration", "documentation", "tutorial",
            "api", "reference", "method", "property", "parameter",
            
            # Specific tools/processes
            "blender", "model", "texture", "material", "animation",
            "weapon", "vehicle", "ai", "damage", "health", "physics",
            "editor", "world editor", "resource manager"
        ]
        
        # Follow-up and clarification words that need more info
        help_keywords = [
            "expand", "expand on", "tell me more", "more details", "elaborate",
            "i didn't understand", "i don't understand", "didn't understand", "don't understand",
            "help me", "can you help", "need help", "help with",
            "explain", "explain more", "explain that", "clarify", "unclear",
            "what does that mean", "what do you mean", "confused", "confusing",
            "show me", "show example", "give example", "demonstrate",
            "more about", "details about", "information about", "how does"
        ]
        
        # Question patterns that usually need docs
        question_patterns = [
            "how do i", "how to", "what is", "where is", "can you show",
            "example of", "tell me about", "can you explain", "configure", "setup",
            "what about", "how about", "what if", "why does", "why is",
            "where can i", "when should i", "which", "best way to"
        ]
        
        # Check for technical keywords
        technical_matches = [kw for kw in technical_keywords if kw in query_lower]
        if technical_matches:
            return True, f"technical content: {', '.join(technical_matches[:3])}"
        
        # Check for help keywords - but be contextual
        help_matches = [kw for kw in help_keywords if kw in query_lower]
        if help_matches:
            # If recent response was technical, help requests likely need docs
            if recent_technical:
                return True, f"follow-up to technical discussion: {', '.join(help_matches[:2])}"
            # General "help" without context - depends on if it's "help with X"
            elif any(pattern in query_lower for pattern in ["help with", "help me with", "need help with"]):
                return True, f"specific help request: {', '.join(help_matches[:2])}"
            # Just "help" by itself might be conversational
            elif query_lower.strip() in ["help", "help me", "i need help"]:
                return False, "general help request without context"
        
        # Check for question patterns
        question_matches = [pattern for pattern in question_patterns if pattern in query_lower]
        if question_matches:
            return True, f"question pattern: {', '.join(question_matches[:2])}"
        
        return False, "conversational query"
    
    def _was_recent_response_technical(self, conversation_context: List) -> bool:
        """Check if the most recent assistant response contained technical content"""
        if not conversation_context:
            return False
        
        # Look at last assistant message
        for msg in reversed(conversation_context):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').lower()
                technical_indicators = [
                    "scr_", "enfusion", "component", "script", "class", "function",
                    "workbench", "import", "export", "bones", "skeleton", "fbx",
                    "weapon", "vehicle", "modding", "code", "error", "configuration"
                ]
                return any(indicator in content for indicator in technical_indicators)
        
        return False

    async def process_chat_message(self, message: ChatMessage) -> EnhancedChatResponse:
        """Main chat processing entry point"""
        # Get or create session
        session_id = self.memory.get_or_create_session(message.session_id)
        logger.info(f"🔧 Smart processing enabled: {message.use_smart_processing}")
        
        # ADDED: Get enhanced context for better understanding
        enhanced_context = self.memory.get_contextual_prompt_enhancement(session_id, message.message)
        
        # CHECK FOR MEMORY+DOC PREFIXES FIRST
        if message.message.strip().lower().startswith('memory+'):
            return await self._handle_memory_plus_doc_query(message, session_id, enhanced_context)
        
        # CHECK FOR MEMORY PREFIX
        if message.message.strip().lower().startswith('memory '):
            return await self._handle_memory_query(message, session_id)
        
        # CHECK FOR REFORM PREFIX
        if message.message.strip().lower().startswith('reform '):
            return await self._handle_reform_query(message, session_id)
        
        # CHECK FOR ANY EXPLICIT PREFIX (including base_search)
        cleaned_question, forced_categories, prefix_used = self.query_processor.detect_category_prefix(message.message)
        
        if prefix_used:
            # User used a prefix - do document search
            logger.info(f"🔧 Prefix detected: '{prefix_used}' - processing with documents")
            if message.use_smart_processing:
                return await self._handle_smart_query(message, session_id, enhanced_context)
            else:
                # Basic processing without smart features
                result = await self._process_query(
                    message.message,
                    session_id,
                    message.max_docs or self.config.max_docs_per_query,
                    message.categories,
                    enhanced_context
                )
                return result
        else:
            # NO PREFIX - Check if conversational query needs document search
            conversation_context = self.memory.sessions.get(session_id, [])
            should_search, reason = self.should_auto_search(message.message, conversation_context)
            
            if should_search:
                logger.info(f"🔍 Auto-detected need for document search: {reason}")
                # Process as if user used base_search prefix
                return await self._process_query(
                    message.message, session_id, 
                    self.config.base_search_max_docs, message.categories, enhanced_context
                )
            else:
                logger.info(f"💬 Staying in conversation mode: {reason}")
                return await self._process_memory_query(message.message, session_id, time.time())

    async def _handle_memory_query(self, message: ChatMessage, session_id: str) -> EnhancedChatResponse:
        """Handle memory-only queries with the 'memory' prefix"""
        logger.info(f"💭 MEMORY QUERY DETECTED: Processing conversation analysis")
        
        # Extract the actual question (remove 'memory ' prefix)
        actual_question = message.message.strip()[7:].strip()
        
        if not actual_question:
            # No question provided after 'memory'
            answer = "Please ask a question about our conversation after 'memory'. For example: 'memory what have we discussed?' or 'memory what was my first question?'"
        else:
            start_time = time.time()
            
            # Check if this is a specific memory analysis request
            if any(word in actual_question.lower() for word in ["analyze", "summarize", "what have we", "topics", "discussed"]):
                answer = self.memory.analyze_conversation_for_memory_query(session_id, actual_question)
            else:
                answer = await self.llm_manager.generate_memory_only_answer(actual_question, session_id, self.memory)
            
            processing_time = time.time() - start_time
        
        # CHANGED: Use add_conversation_turn for memory queries too
        if not actual_question:
            processing_time = 0.1
        self.memory.add_conversation_turn(session_id, message.message, answer)
        
        # Get conversation summary
        conv_summary = self.memory.get_conversation_summary(session_id)
        
        # Return memory-specific response
        enhanced_result = EnhancedChatResponse(
            question=message.message,
            answer=answer,
            session_id=session_id,
            context_used=0,  # No documents used
            sources=[],
            processing_time=getattr(locals(), 'processing_time', 0.1),
            timestamp=datetime.now().isoformat(),
            conversation_length=conv_summary["length"],
            query_type="memory_query",
            processed_query=actual_question,
            suggested_follow_ups=[
                "memory what else have we discussed?", 
                "memory can you summarize our conversation?",
                "Ask a new question about Arma Reforger"
            ],
            confidence_score=1.0,
            multi_part=False,
            corrections_made=[],
            category_prefix_used="memory",
            forced_categories=[],
            intent_detected="memory_analysis",
            context_enhanced=True,
            validation_issues=[]
        )
        logger.info(f"✅ Memory query completed - analyzing conversation history only")
        return enhanced_result

    async def _handle_memory_plus_doc_query(self, message: ChatMessage, session_id: str, enhanced_context: str) -> EnhancedChatResponse:
        """Handle memory+doc hybrid queries - memory first, docs if needed"""
        logger.info(f"🧠📚 MEMORY+DOC QUERY: Processing hybrid memory+document analysis")
        
        # Parse the prefix to get categories
        prefix_parts = message.message.strip().lower().split(' ', 1)
        prefix_used = prefix_parts[0]  # e.g., "memory+doc"
        actual_question = prefix_parts[1] if len(prefix_parts) > 1 else ""
        
        if not actual_question:
            answer = f"Please ask a question after '{prefix_used}'. For example: '{prefix_used} expand on the World Editor Plugin'"
            return self._create_memory_response(message, session_id, answer, prefix_used)
        
        start_time = time.time()
        
        # STEP 1: Try memory-only analysis first
        logger.info("🧠 Step 1: Analyzing conversation memory...")
        
        try:
            # Check if we have good conversation context for this question
            if any(word in actual_question.lower() for word in ["expand", "more about", "explain", "detail"]):
                memory_answer = self.memory.analyze_conversation_for_memory_query(session_id, actual_question)
            else:
                memory_answer = await self.llm_manager.generate_memory_only_answer(actual_question, session_id, self.memory)
            
            # STEP 2: Assess memory answer quality
            memory_confidence = await self._assess_memory_answer_quality(memory_answer, actual_question)
            logger.info(f"🎯 Memory answer confidence: {memory_confidence:.2f}")
            
            # STEP 3: If memory confidence is high enough, use memory answer
            if memory_confidence >= 0.7:
                logger.info("✅ High confidence memory answer - using memory only")
                final_answer = f"**Based on our conversation:**\n\n{memory_answer}"
                processing_time = time.time() - start_time
                
                self.memory.add_conversation_turn(session_id, message.message, final_answer)
                return self._create_memory_response(message, session_id, final_answer, prefix_used, processing_time, context_used=0)
            
            # STEP 4: Low confidence - supplement with documents
            logger.info("📚 Low memory confidence - supplementing with document search")
            
            # Get categories from prefix
            prefix_categories = {
                "memory+doc": ["Documentation"],
                "memory+code": ["Source_Code"],
                "memory+api": ["API_Reference"],
                "memory+all": ["Source_Code", "API_Reference", "Documentation"]
            }
            categories = prefix_categories.get(prefix_used, ["Documentation"])
            max_docs = 30  # Moderate document search for supplementation
            
            # Retrieve documents
            relevant_categories = self._determine_relevant_categories(actual_question, categories)
            relevant_docs = await self._retrieve_documents_smart(actual_question, max_docs, relevant_categories, [])
            
            # STEP 5: Generate hybrid answer (memory + documents)
            hybrid_answer = await self._generate_memory_plus_doc_answer(
                actual_question, memory_answer, relevant_docs, session_id, enhanced_context
            )
            
            processing_time = time.time() - start_time
            self.memory.add_conversation_turn(session_id, message.message, hybrid_answer)
            
            return self._create_memory_response(
                message, session_id, hybrid_answer, prefix_used, processing_time, 
                context_used=len(relevant_docs), sources=relevant_docs
            )
            
        except Exception as e:
            logger.error(f"❌ Memory+Doc processing failed: {e}")
            # Fallback to regular document search
            return await self._process_query(message.message, session_id, 30, categories, enhanced_context)
    
    async def _assess_memory_answer_quality(self, memory_answer: str, question: str) -> float:
        """Assess the quality of memory-only answer"""
        
        # Quick heuristic assessment
        confidence = 0.5  # Base confidence
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "not explicitly detailed", "not covered", "uncertainty", "limitations",
            "may be necessary", "further exploration", "not specifically mentioned",
            "general role", "basic", "lacks depth"
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase.lower() in memory_answer.lower())
        confidence -= uncertainty_count * 0.15  # Reduce confidence for uncertainty
        
        # Check for specific details
        specific_indicators = [
            "step", "example", "code", "function", "method", "process", "procedure",
            "workflow", "implementation", "configuration", "setup"
        ]
        
        specificity_count = sum(1 for indicator in specific_indicators if indicator.lower() in memory_answer.lower())
        confidence += specificity_count * 0.1  # Increase confidence for specificity
        
        # Check answer length (very short answers usually lack detail)
        if len(memory_answer) < 200:
            confidence -= 0.2
        elif len(memory_answer) > 500:
            confidence += 0.1
        
        # Check if answer directly addresses the question
        question_words = set(question.lower().split())
        answer_words = set(memory_answer.lower().split())
        overlap = len(question_words.intersection(answer_words)) / len(question_words)
        confidence += overlap * 0.2
        
        return max(0.0, min(1.0, confidence))
    
    async def _generate_memory_plus_doc_answer(self, question: str, memory_answer: str, 
                                             docs: List, session_id: str, enhanced_context: str) -> str:
        """Generate hybrid answer combining memory and document insights"""
        
        # Create hybrid prompt that combines memory insights with document evidence
        hybrid_prompt = f"""You are an expert Arma Reforger analyst. You have conversation context and new documentation to provide a comprehensive answer.

CONVERSATION CONTEXT:
{enhanced_context}

PREVIOUS MEMORY ANALYSIS:
{memory_answer}

NEW DOCUMENTATION EVIDENCE:
{self._format_documents_for_hybrid(docs)}

USER QUESTION: {question}

HYBRID ANALYSIS INSTRUCTIONS:
1. **Acknowledge** what we discussed in our conversation
2. **Identify** gaps or limitations in the memory analysis
3. **Supplement** with specific information from the documentation
4. **Combine** both sources for a comprehensive answer
5. **Highlight** new information that wasn't available in our conversation

Provide a complete answer that builds on our conversation and adds new documentation insights.

Answer:"""
        
        return await self.llm_manager.call_ollama(hybrid_prompt)
    
    def _format_documents_for_hybrid(self, docs: List) -> str:
        """Format documents for hybrid memory+doc analysis"""
        if not docs:
            return "No additional documentation found."
        
        formatted_docs = []
        for i, doc in enumerate(docs[:5], 1):  # Limit to 5 docs for hybrid
            filename = doc.metadata.get('filename', 'Unknown')
            category = doc.metadata.get('main_category', 'Unknown')
            content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
            
            formatted_docs.append(f"Document {i} - {filename} [{category}]:\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    def _create_memory_response(self, message: ChatMessage, session_id: str, answer: str, 
                              prefix_used: str, processing_time: float = 0.1, 
                              context_used: int = 0, sources: List = None) -> EnhancedChatResponse:
        """Create standardized memory response"""
        
        conv_summary = self.memory.get_conversation_summary(session_id)
        
        if sources:
            formatted_sources = [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "category": f"{doc.metadata.get('main_category', 'Unknown')}/{doc.metadata.get('sub_category', 'Unknown')}",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in sources
            ]
        else:
            formatted_sources = []
        
        # Determine query type
        if prefix_used.startswith("memory+"):
            query_type = "memory_plus_doc_query"
            suggested_follow_ups = [
                f"memory+doc tell me more about this topic",
                "memory what other topics have we covered?",
                "Ask a new technical question"
            ]
        else:
            query_type = "memory_query"
            suggested_follow_ups = [
                "memory what else have we discussed?", 
                "memory can you summarize our conversation?",
                "Ask a new question about Arma Reforger"
            ]
        
        return EnhancedChatResponse(
            question=message.message,
            answer=answer,
            session_id=session_id,
            context_used=context_used,
            sources=formatted_sources,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            conversation_length=conv_summary["length"],
            query_type=query_type,
            processed_query=message.message.split(' ', 1)[1] if ' ' in message.message else message.message,
            suggested_follow_ups=suggested_follow_ups,
            confidence_score=1.0,
            multi_part=False,
            corrections_made=[],
            category_prefix_used=prefix_used,
            forced_categories=[],
            intent_detected="memory_analysis" if not prefix_used.startswith("memory+") else "memory_plus_doc_analysis",
            context_enhanced=True,
            validation_issues=[]
        )

    async def _handle_reform_query(self, message: ChatMessage, session_id: str) -> EnhancedChatResponse:
        """Handle AI-powered question reform"""
        logger.info(f"🤖 REFORM DETECTED: Processing AI-powered question improvement")
        
        # Extract the actual question (remove 'reform ' prefix)
        actual_question = message.message.strip()[7:].strip()
        
        # Use AI to reform the question
        start_time = time.time()
        reform_response = await self.llm_manager.reform_question_with_ai(actual_question, session_id, self.memory)
        processing_time = time.time() - start_time
        
        # CHANGED: Use add_conversation_turn instead of separate add_message calls
        self.memory.add_conversation_turn(session_id, message.message, reform_response)
        
        # Get conversation summary
        conv_summary = self.memory.get_conversation_summary(session_id)
        
        # Return AI-powered reform response
        enhanced_result = EnhancedChatResponse(
            question=message.message,
            answer=reform_response,
            session_id=session_id,
            context_used=0,  # No documents used
            sources=[],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            conversation_length=conv_summary["length"],
            query_type="ai_reform",
            processed_query=actual_question,
            suggested_follow_ups=[
                "Try one of the improved questions above",
                "How do I ask better Arma Reforger questions?", 
                "What makes a good technical question?"
            ],
            confidence_score=1.0,
            multi_part=False,
            corrections_made=[],
            category_prefix_used="reform",
            forced_categories=[],
            intent_detected="ai_reform_request",
            context_enhanced=bool(self.memory.get_contextual_prompt_enhancement(session_id, message.message)),
            validation_issues=[]
        )
        logger.info(f"✅ AI Reform completed in {processing_time:.2f}s - 0 documents used")
        return enhanced_result
    
    async def _handle_smart_query(self, message: ChatMessage, session_id: str, enhanced_context: str = "") -> EnhancedChatResponse:
        """Handle smart processing with prefixes and intelligence"""
        # Check for category prefixes
        cleaned_question, forced_categories, prefix_used = self.query_processor.detect_category_prefix(message.message)
        logger.info(f"🔧 Prefix detection: '{prefix_used}' | Categories: {forced_categories}")
        
        # Determine max_docs based on prefix
        override_max_docs = self._determine_max_docs_from_prefix(prefix_used, message.max_docs)
        
        # Process with smart features
        categories = message.categories
        if forced_categories:
            categories = forced_categories
            question_to_process = cleaned_question
        else:
            question_to_process = message.message
        
        # Get conversation history for intent detection
        conversation_history = self.memory.sessions.get(session_id, [])
        
        # Intent detection and query processing
        try:
            intent_result = self.query_processor.detect_intent(question_to_process, conversation_history)
            logger.info(f"🎯 Intent: {intent_result['intent_type']}")
        except Exception as e:
            logger.error(f"Error in intent detection: {e}")
            intent_result = {"intent_type": "direct_question", "validation_issues": []}
        
        # Query classification and corrections
        query_type = self.query_classifier.classify_query(question_to_process)
        processed_query, corrections_made = self.query_processor.correct_and_expand(question_to_process)
        if corrections_made:
            logger.info(f"✏️ Query corrections: {corrections_made}")
        
        # Route to appropriate processing method
        result = await self._route_query_processing(
            question_to_process, session_id, override_max_docs, categories, prefix_used, enhanced_context
        )
        
        # Convert to EnhancedChatResponse
        enhanced_result = EnhancedChatResponse(
            question=result.question,
            answer=result.answer,
            session_id=result.session_id,
            context_used=result.context_used,
            sources=result.sources,
            processing_time=result.processing_time,
            timestamp=result.timestamp,
            conversation_length=result.conversation_length,
            query_type=query_type,
            processed_query=processed_query,
            suggested_follow_ups=[],
            confidence_score=0.9,
            multi_part=False,
            corrections_made=corrections_made,
            category_prefix_used=prefix_used if prefix_used else None,
            forced_categories=forced_categories,
            intent_detected=intent_result["intent_type"],
            context_enhanced=bool(enhanced_context),
            validation_issues=intent_result["validation_issues"]
        )
        return enhanced_result
    
    def _determine_max_docs_from_prefix(self, prefix_used: str, message_max_docs: Optional[int]) -> int:
        """Determine max_docs based on detected prefix"""
        if not prefix_used:
            return message_max_docs or self.config.max_docs_per_query
        
        # Base search prefix - explicit document search
        if prefix_used == "base_search":
            max_docs = self.config.base_search_max_docs
            logger.info(f"🔍 BASE_SEARCH: using {max_docs} documents (explicit search)")
            return max_docs
        
        # Quick prefixes
        if prefix_used.startswith("quick_"):
            prefix_map = {
                "quick_doc": self.config.quick_doc_max_docs,
                "quick_code": self.config.quick_code_max_docs,
                "quick_api": self.config.quick_api_max_docs,
                "quick_all": self.config.quick_all_max_docs,
            }
            max_docs = prefix_map.get(prefix_used, 5)
            logger.info(f"⚡ {prefix_used.upper()}: using {max_docs} documents (fast)")
            
        # Standard prefixes
        elif prefix_used.startswith("standard_"):
            prefix_map = {
                "standard_doc": self.config.standard_doc_max_docs,
                "standard_code": self.config.standard_code_max_docs,
                "standard_api": self.config.standard_api_max_docs,
                "standard_code+api": self.config.standard_code_api_max_docs,
                "standard_all": self.config.standard_all_max_docs,
            }
            max_docs = prefix_map.get(prefix_used, 50)
            logger.info(f"🎯 {prefix_used.upper()}: using {max_docs} documents")
            
        # Force prefixes
        elif prefix_used.startswith("force_"):
            prefix_map = {
                "force_doc": self.config.force_doc_max_docs,
                "force_code": self.config.force_code_max_docs,
                "force_api": self.config.force_api_max_docs,
                "force_code+api": self.config.force_code_api_max_docs,
                "force_all": self.config.force_all_max_docs,
                "force_benchmark": self.config.force_benchmark_max_docs,
            }
            max_docs = prefix_map.get(prefix_used, 500)
            logger.info(f"🚀 {prefix_used.upper()}: using {max_docs} documents")
            
        # Dynamic prefixes
        elif prefix_used.startswith("dynamic_"):
            prefix_map = {
                "dynamic_doc": self.config.dynamic_doc_max_docs,
                "dynamic_code": self.config.dynamic_code_max_docs,
                "dynamic_api": self.config.dynamic_api_max_docs,
                "dynamic_code+api": self.config.dynamic_code_api_max_docs,
                "dynamic_all": self.config.dynamic_all_max_docs,
            }
            max_docs = prefix_map.get(prefix_used, 50)
            logger.info(f"🧠 {prefix_used.upper()}: using progressive up to {max_docs} documents")
            
        else:
            max_docs = message_max_docs or self.config.max_docs_per_query
            logger.info(f"🔧 DEFAULT: no prefix detected - using {max_docs} documents")
        
        return max_docs
    
    async def _route_query_processing(self, question: str, session_id: str, max_docs: int, 
                                    categories: Optional[List[str]], prefix_used: str, enhanced_context: str = "") -> EnhancedChatResponse:
        """Route to appropriate processing method based on prefix type"""
        dynamic_prefixes = ["dynamic_doc", "dynamic_code", "dynamic_api", "dynamic_code+api", "dynamic_all"]
        quick_prefixes = ["quick_doc", "quick_code", "quick_api", "quick_all"]
        
        if prefix_used in dynamic_prefixes:
            logger.info(f"🧠 Using progressive retrieval for '{prefix_used}'")
            return await self._progressive_retrieval(question, session_id, max_docs, categories, prefix_used, enhanced_context)
        elif prefix_used in quick_prefixes:
            logger.info(f"⚡ Using quick retrieval for '{prefix_used}'")
            return await self._process_query_quick(question, session_id, max_docs, categories, enhanced_context)
        else:
            logger.info(f"⚡ Using standard retrieval")
            return await self._process_query(question, session_id, max_docs, categories, enhanced_context)
    
    async def _process_query(self, question: str, session_id: str, max_docs: Optional[int] = None, 
                           categories: Optional[List[str]] = None, enhanced_context: str = "") -> EnhancedChatResponse:
        """Process a basic query"""
        start_time = time.time()
        
        # Check if this is a memory-only query
        is_memory_only = question.startswith("MEMORY_ONLY:") or max_docs == 0
        
        if is_memory_only:
            return await self._process_memory_query(question, session_id, start_time)
        
        # Regular RAG query with document search
        max_docs = max_docs or self.config.max_docs_per_query
        logger.info(f"🔧 Processing query with max_docs={max_docs}")
        
        # Determine relevant categories and retrieve documents
        relevant_categories = self._determine_relevant_categories(question, categories)
        logger.info(f"🔍 Searching in categories: {relevant_categories}")
        
        relevant_docs = await self._retrieve_documents_smart(question, max_docs, relevant_categories, [])
        
        # CHANGED: Generate answer with enhanced context
        answer = await self.llm_manager.generate_answer(question, relevant_docs, session_id, self.memory, enhanced_context)
        
        # CHANGED: Use add_conversation_turn instead of separate add_message calls
        self.memory.add_conversation_turn(session_id, question, answer)
        
        # Get conversation summary
        conv_summary = self.memory.get_conversation_summary(session_id)
        
        # Return EnhancedChatResponse
        response = EnhancedChatResponse(
            question=question,
            answer=answer,
            session_id=session_id,
            context_used=len(relevant_docs),
            sources=[
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "category": f"{doc.metadata.get('main_category', 'Unknown')}/{doc.metadata.get('sub_category', 'Unknown')}",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in relevant_docs
            ],
            processing_time=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            conversation_length=conv_summary["length"],
            query_type="regular_query",
            processed_query=question,
            suggested_follow_ups=["Can you expand on that?", "What about related topics?", "Show me more examples"],
            confidence_score=0.8,
            multi_part=False,
            corrections_made=[],
            category_prefix_used=None,
            forced_categories=categories,
            intent_detected="direct_question",
            context_enhanced=bool(enhanced_context),
            validation_issues=[]
        )
        
        logger.info(f"✅ Query processed in {response.processing_time:.2f}s")
        return response
    
    async def _process_memory_query(self, question: str, session_id: str, start_time: float) -> EnhancedChatResponse:
        """Process memory-only query - ENHANCED VERSION"""
        if question.startswith("MEMORY_ONLY:"):
            question = question[12:].strip()
        
        try:
            # CHANGED: Check if this is a specific memory analysis request
            if any(word in question.lower() for word in ["analyze", "summarize", "what have we", "topics", "discussed"]):
                answer = self.memory.analyze_conversation_for_memory_query(session_id, question)
            else:
                answer = await self.llm_manager.generate_memory_only_answer(question, session_id, self.memory)
            
            # Check if we got a timeout error
            if "taking too long" in answer.lower():
                # Fallback to simple conversation response
                answer = "I remember our conversation, but I'm having trouble processing that right now. Could you rephrase the question or ask something more specific?"
                
        except Exception as e:
            logger.error(f"Memory query failed: {e}")
            # Simple fallback response
            answer = "I'm having trouble accessing our conversation history right now. Could you rephrase your question?"
        
        # CHANGED: Use add_conversation_turn
        self.memory.add_conversation_turn(session_id, question, answer)
        conv_summary = self.memory.get_conversation_summary(session_id)
        
        response = EnhancedChatResponse(
            question=question,
            answer=answer,
            session_id=session_id,
            context_used=0,
            sources=[],
            processing_time=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            conversation_length=conv_summary["length"],
            query_type="memory_query",
            processed_query=question,
            suggested_follow_ups=["What else have we discussed?", "Can you summarize our conversation?", "Ask about Arma Reforger instead"],
            confidence_score=1.0,
            multi_part=False,
            corrections_made=[],
            category_prefix_used=None,
            forced_categories=[],
            intent_detected="memory_query",
            context_enhanced=True,  # Memory queries always use enhanced context
            validation_issues=[]
        )
        
        logger.info(f"✅ Memory-only query processed in {response.processing_time:.2f}s")
        return response
    
    async def _process_query_quick(self, question: str, session_id: str, max_docs: Optional[int] = None, 
                                 categories: Optional[List[str]] = None, enhanced_context: str = "") -> EnhancedChatResponse:
        """Process a quick query with optimized settings for speed"""
        start_time = time.time()
        
        max_docs = max_docs or 5
        logger.info(f"⚡ Quick processing with max_docs={max_docs}")
        
        # Determine relevant categories (prioritize most relevant only)
        relevant_categories = self._determine_relevant_categories(question, categories)[:1]  # Only use top category
        logger.info(f"⚡ Quick search in category: {relevant_categories}")
        
        # Quick document retrieval
        relevant_docs = await self._retrieve_documents_quick(question, max_docs, relevant_categories)
        
        # CHANGED: Generate answer with enhanced context
        answer = await self.llm_manager.generate_answer_quick(question, relevant_docs, session_id, self.memory, enhanced_context)
        
        # CHANGED: Use add_conversation_turn
        self.memory.add_conversation_turn(session_id, question, answer)
        
        # Get conversation summary
        conv_summary = self.memory.get_conversation_summary(session_id)
        
        # Prepare enhanced response
        response = EnhancedChatResponse(
            question=question,
            answer=answer,
            session_id=session_id,
            context_used=len(relevant_docs),
            sources=[
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "category": f"{doc.metadata.get('main_category', 'Unknown')}/{doc.metadata.get('sub_category', 'Unknown')}",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in relevant_docs
            ],
            processing_time=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            conversation_length=conv_summary["length"],
            query_type="quick_query",
            processed_query=question,
            suggested_follow_ups=["Can you expand on that?", "Show me more examples", "What about related topics?"],
            confidence_score=0.8,
            multi_part=False,
            corrections_made=[],
            category_prefix_used=None,
            forced_categories=categories,
            intent_detected="direct_question",
            context_enhanced=bool(enhanced_context),
            validation_issues=[]
        )
        
        logger.info(f"⚡ Quick query processed in {response.processing_time:.2f}s")
        return response
    
    async def _progressive_retrieval(self, question: str, session_id: str, max_docs: int, 
                                   categories: Optional[List[str]], prefix_used: str = None, enhanced_context: str = "") -> EnhancedChatResponse:
        """Implement progressive retrieval with LLM confidence assessment"""
        start_time = time.time()
        
        logger.info(f"🧠 Progressive retrieval (max: {max_docs} docs)")
        
        current_batch = self.config.progressive_initial_batch
        
        for round_num in range(1, self.config.progressive_max_expansions + 2):
            logger.info(f"📄 Round {round_num}: Retrieving {current_batch} documents")
            
            relevant_categories = self._determine_relevant_categories(question, categories)
            round_docs = await self._retrieve_documents_smart(question, current_batch, relevant_categories, [])
            
            # CHANGED: Generate answer with enhanced context
            answer = await self.llm_manager.generate_answer(question, round_docs, session_id, self.memory, enhanced_context)
            confidence = await self.llm_manager.assess_answer_confidence(question, answer, round_docs)
            
            logger.info(f"🎯 Round {round_num}: {len(round_docs)} docs, confidence: {confidence:.2f}")
            
            if confidence >= self.config.progressive_expand_threshold or current_batch >= max_docs:
                # CHANGED: Use add_conversation_turn
                self.memory.add_conversation_turn(session_id, question, answer)
                conv_summary = self.memory.get_conversation_summary(session_id)
                
                return EnhancedChatResponse(
                    question=question,
                    answer=answer,
                    session_id=session_id,
                    context_used=len(round_docs),
                    sources=[
                        {
                            "filename": doc.metadata.get("filename", "Unknown"),
                            "category": f"{doc.metadata.get('main_category', 'Unknown')}/{doc.metadata.get('sub_category', 'Unknown')}",
                            "source": doc.metadata.get("source", "Unknown")
                        }
                        for doc in round_docs
                    ],
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat(),
                    conversation_length=conv_summary["length"],
                    query_type="dynamic_query",
                    processed_query=question,
                    suggested_follow_ups=["Can you expand on that?", "Show me more examples", "What about alternatives?"],
                    confidence_score=confidence,
                    multi_part=False,
                    corrections_made=[],
                    category_prefix_used=prefix_used,
                    forced_categories=categories,
                    intent_detected="direct_question",
                    context_enhanced=bool(enhanced_context),
                    validation_issues=[]
                )
            
            if round_num <= self.config.progressive_max_expansions:
                next_batch = int(current_batch * self.config.progressive_expansion_multiplier)
                current_batch = min(next_batch, max_docs)
                logger.info(f"🔄 Expanding to {current_batch} docs")
            else:
                logger.info(f"⚠️ Max expansions reached")
                break
        
        logger.warning("⚠️ Progressive retrieval fallback")
        return await self._process_query(question, session_id, max_docs, categories, enhanced_context)
    
    def _determine_relevant_categories(self, query: str, categories: Optional[List[str]] = None) -> List[str]:
        """Determine which document categories are most relevant"""
        if categories:
            return [cat for cat in categories if cat in self.vector_stores]
        
        query_lower = query.lower()
        
        category_keywords = {
            "Documentation": [
                "import", "imported", "importing", "missing", "broken", "not working",
                "how to", "tutorial", "guide", "getting started", "setup", "installation", 
                "workflow", "process", "steps", "example", "basics", "workbench", "editor",
                "modding", "create", "make", "build", "configure", "problem", "issue",
                "why", "error", "fix", "solution", "trouble", "help", "can't", "won't",
                "doesn't work", "not showing", "attachment points", "bones missing"
            ],
            "Source_Code": [
                "code", "function", "class", "method", "implementation", "script",
                "variable", "programming", "coding", "algorithm", "scr_", "enfusion",
                "component", "system"
            ],
            "API_Reference": [
                "api", "reference", "interface", "class reference", "function reference",
                "parameter", "return", "inherit", "method", "property", "documentation",
                "specification", "definition"
            ]
        }
        
        category_scores = {}
        for category, keywords in category_keywords.items():
            if category in self.vector_stores:
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    category_scores[category] = score
        
        # Boost Documentation for troubleshooting
        troubleshooting_keywords = ["missing", "broken", "not working", "error", "problem", "issue", "why", "can't", "won't", "doesn't work", "imported", "import"]
        has_troubleshooting = any(keyword in query_lower for keyword in troubleshooting_keywords)
        
        if has_troubleshooting and "Documentation" in category_scores:
            category_scores["Documentation"] += 5
            logger.info(f"🔧 Troubleshooting detected - boosting Documentation")
        
        # For short queries, search all
        query_words = query_lower.split()
        if len(query_words) <= 2 or query_lower in ["script", "code", "function", "component", "weapon", "ai", "vehicle"]:
            logger.info(f"🔍 Short/generic query - searching all categories")
            return list(self.vector_stores.keys())
        
        if category_scores:
            sorted_categories = [cat for cat, _ in sorted(category_scores.items(), key=lambda x: x[1], reverse=True)]
            logger.info(f"🎯 Category scores: {category_scores} → Selected: {sorted_categories}")
            return sorted_categories
        else:
            logger.info(f"🔍 No keywords found - searching all categories")
            return list(self.vector_stores.keys())
    
    async def _retrieve_documents_smart(self, query: str, max_docs: int, preferred_categories: Optional[List[str]], boost_keywords: List[str]) -> List[Document]:
        """Retrieve documents with smart category prioritization"""
        return await self.llm_manager.retrieve_documents_smart(query, max_docs, preferred_categories, boost_keywords)
    
    async def _retrieve_documents_quick(self, query: str, max_docs: int, categories: List[str]) -> List[Document]:
        """Quick document retrieval - simplified and fast"""
        return await self.llm_manager.retrieve_documents_quick(query, max_docs, categories)
    
    # Conversation management methods
    async def get_conversation(self, session_id: str):
        """Get conversation history for a session"""
        summary = self.memory.get_conversation_summary(session_id)
        return {
            "session_id": session_id,
            "summary": summary,
            "messages": self.memory.sessions.get(session_id, [])
        }
    
    async def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        self.memory.clear_session(session_id)
    
    async def get_system_stats(self) -> SystemStats:
        """Get system statistics"""
        return await self.llm_manager.get_system_stats(self.memory)
    
    async def health_check(self):
        """System health check"""
        return await self.llm_manager.health_check(self.memory)