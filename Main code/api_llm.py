import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain imports
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma
    
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
    
from langchain.schema import Document
import requests

from config import APIConfig
from models import SystemStats

# DYNAMIC AI BEHAVIOR - No hardcoded solutions!
from ai_behavior import DynamicAIBehaviorCoordinator

logger = logging.getLogger(__name__)

class LLMManager:
    """Handles all LLM communication, system initialization, vector store management, and dynamic AI behavior coordination"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.embeddings = None
        self.vector_stores = {}
        self.retrievers = {}
        self.system_ready = False
        
        # Initialize system
        self._initialize_system()
        
        # DYNAMIC: Initialize AI behavior coordinator (no hardcoded solutions)
        logger.info("🤖 Initializing Dynamic AI Behavior Coordinator...")
        self.ai_behavior = DynamicAIBehaviorCoordinator()
        logger.info("✅ Dynamic AI Behavior Coordinator ready for intelligent analysis!")
    
    def _initialize_system(self):
        """Initialize the RAG system"""
        logger.info("🚀 Initializing Arma RAG API...")
        
        try:
            # Setup embeddings
            logger.info("📚 Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            
            # Load existing vector stores
            logger.info("🗃️ Loading vector stores...")
            self._load_vector_stores()
            
            if self.vector_stores:
                self.system_ready = True
                logger.info("✅ System ready!")
            else:
                logger.warning("⚠️ No vector stores found. Please run document processing first.")
                
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
    
    def _load_vector_stores(self):
        """Load existing vector stores"""
        for category in ["Source_Code", "Documentation", "API_Reference"]:
            collection_name = f"arma_{category.lower()}"
            
            try:
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.config.vector_db_path
                )
                
                # Check if collection has documents
                count = vector_store._collection.count()
                if count > 0:
                    self.vector_stores[category] = vector_store
                    
                    # Create retriever with config values
                    retriever_k = self.config.max_docs_per_query + 10
                    fetch_k = self.config.max_docs_per_query * 3
                    
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": retriever_k,
                            "fetch_k": fetch_k,
                            "lambda_mult": 0.5,
                        }
                    )
                    self.retrievers[category] = retriever
                    
                    logger.info(f"✅ Loaded {category}: {count} documents")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not load {category}: {e}")
                try:
                    fallback_k = self.config.max_docs_per_query * 2
                    retriever = vector_store.as_retriever(search_kwargs={"k": fallback_k})
                    self.retrievers[category] = retriever
                    logger.info(f"✅ Loaded {category}: {count} documents (fallback mode)")
                except Exception as e2:
                    logger.error(f"❌ Failed to create retriever for {category}: {e2}")
    
    # LLM Communication Methods
    async def call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation (standard settings)"""
        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_ctx": 4096  # Full context for comprehensive responses
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Could not get response from AI model."
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error: {str(e)}"
    
    async def call_ollama_fast(self, prompt: str) -> str:
        """Call Ollama API with optimized settings for fast responses"""
        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_ctx": 2048,  # Reduced context for speed
                        "num_predict": -1  # Fixed cutoff issue
                    }
                },
                timeout=180  # Reasonable timeout
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Could not get response from AI model."
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Sorry, the AI is taking too long. Try asking a simpler question or restart the system."
    
    # ENHANCED: Answer Generation Methods with Enhanced Memory Context
    async def generate_answer(self, query: str, context_docs: List[Document], session_id: str, memory, enhanced_context: str = "") -> str:
        """Generate answer using dynamic AI behavior coordination with enhanced memory context"""
        
        if not context_docs:
            return "I couldn't find relevant information in the Arma Reforger documentation to answer your question."
        
        # Check if this query needs intelligent analysis
        if self.ai_behavior.should_use_intelligent_analysis(query):
            logger.info("🧠 Using dynamic AI behavior analysis - discovering solutions from evidence")
            
            # ENHANCED: Use enhanced conversation context if available
            if enhanced_context:
                conversation_context = enhanced_context
                logger.info("💡 Using enhanced conversation context for better understanding")
            else:
                conversation_context = memory.get_conversation_context(session_id, max_exchanges=3)
            
            # Coordinate dynamic AI analysis and evidence-based prompt building
            analysis_result = self.ai_behavior.analyze_and_respond(
                query, context_docs, conversation_context, response_mode="reasoning"
            )
            
            # Use the intelligently built evidence-based prompt
            prompt = analysis_result["prompt"]
            
            # FIXED: Log the dynamic analysis for debugging with compatibility
            try:
                metadata = analysis_result.get("analysis_metadata", {})
                if metadata:
                    problem_type = metadata.get("problem_type", metadata.get("query_intent", "unknown"))
                    solution_count = metadata.get("solution_candidates", 0)
                    reasoning_steps = metadata.get("reasoning_steps", 0)
                    
                    logger.info(f"🎯 Dynamic Analysis: {problem_type} problem type")
                    logger.info(f"🔍 Evidence: Found {solution_count} solution candidates") 
                    logger.info(f"🧠 Reasoning: {reasoning_steps} reasoning steps")
                    
                    if "reasoning_quality" in analysis_result:
                        logger.info(f"📊 Quality: {analysis_result['reasoning_quality']:.2f} reasoning score")
                    
                    approach = metadata.get("approach", "analysis")
                    logger.info(f"💡 Approach: {approach}")
                else:
                    logger.info("🎯 Analysis completed successfully")
            except Exception as e:
                logger.warning(f"⚠️ Logging error (non-critical): {e}")
            
            return await self.call_ollama(prompt)
        
        else:
            # Use standard analytical generation for non-technical queries
            logger.info("📝 Using standard analytical answer generation")
            return await self._generate_standard_analytical_answer(query, context_docs, session_id, memory, enhanced_context)
    
    async def _generate_standard_analytical_answer(self, query: str, context_docs: List[Document], session_id: str, memory, enhanced_context: str = "") -> str:
        """Generate standard analytical answer with enhanced memory context"""
        
        # ENHANCED: Use enhanced context if available
        if enhanced_context:
            conversation_context = enhanced_context
            logger.info("💡 Using enhanced context for standard analytical answer")
        else:
            conversation_context = memory.get_conversation_context(session_id, max_exchanges=3)
        
        # Use standard analytical prompt builder
        prompt = self.ai_behavior.prompt_builder.build_standard_prompt(query, context_docs, conversation_context)
        
        return await self.call_ollama(prompt)
    
    async def generate_answer_quick(self, query: str, context_docs: List[Document], session_id: str, memory, enhanced_context: str = "") -> str:
        """Generate quick answer with enhanced memory context when available"""
        
        if not context_docs:
            return "I couldn't find relevant information for this quick query. Try a more detailed question or use a standard prefix."
        
        # Check if technical query that needs intelligent analysis
        if self.ai_behavior.should_use_intelligent_analysis(query):
            logger.info("⚡ Using dynamic AI behavior for quick technical analysis")
            
            # ENHANCED: Use enhanced context for quick analysis
            if enhanced_context:
                conversation_context = enhanced_context
                logger.info("💡 Using enhanced context for quick analysis")
            else:
                conversation_context = memory.get_conversation_context(session_id, max_exchanges=1)
            
            # Use quick analysis mode
            analysis_result = self.ai_behavior.analyze_and_respond(
                query, context_docs, conversation_context, response_mode="quick_analysis"
            )
            
            prompt = analysis_result["prompt"]
            
            # FIXED: Log quick dynamic analysis with compatibility
            try:
                confidence = analysis_result.get("analysis_confidence", 0.0)
                evidence_strength = analysis_result.get("evidence_strength", 0.0)
                mode = analysis_result.get("analysis_mode", "unknown")
                
                logger.info(f"⚡ Quick Analysis: {confidence:.2f} confidence, {evidence_strength:.2f} evidence strength, {mode} mode")
            except Exception as e:
                logger.warning(f"⚠️ Quick logging error (non-critical): {e}")
            
            return await self.call_ollama_fast(prompt)
        
        else:
            # Use simple quick response for non-technical queries
            logger.info("⚡ Using simple quick analytical response")
            return await self._generate_simple_quick_answer(query, context_docs, session_id, memory, enhanced_context)
    
    async def _generate_simple_quick_answer(self, query: str, context_docs: List[Document], session_id: str, memory, enhanced_context: str = "") -> str:
        """Generate simple quick answer with enhanced memory context"""
        
        # ENHANCED: Use enhanced context if available
        if enhanced_context:
            conversation_context = enhanced_context
            logger.info("💡 Using enhanced context for simple quick answer")
        else:
            conversation_context = memory.get_conversation_context(session_id, max_exchanges=1)
        
        # Simplified context preparation
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Only use first 3 docs
            context_parts.append(f"Doc {i+1}: {doc.page_content[:300]}...")  # Truncate for speed
        
        rag_context = "\n".join(context_parts)
        
        # Analytical quick prompt
        prompt = f"""You are an expert Arma Reforger analyst. Provide a focused, analytical answer.

{conversation_context}

Context for Analysis:
{rag_context}

Question: {query}

ANALYTICAL APPROACH:
1. Examine the question and context
2. Identify the most relevant information
3. Provide a practical, evidence-based answer

Give a focused answer based on analysis of the available information.

Answer:"""

        return await self.call_ollama_fast(prompt)
    
    async def generate_memory_only_answer(self, query: str, session_id: str, memory) -> str:
        """Generate answer using enhanced conversation memory with analytical approach"""
        
        # ENHANCED: Use enhanced conversation context method
        conversation_context = memory.get_enhanced_conversation_context(session_id, query)
        
        if not conversation_context or "No conversation history available" in conversation_context:
            return "We haven't discussed any topics yet in this conversation. Feel free to ask me about Arma Reforger modding!"
        
        # Get raw messages for additional verification
        raw_messages = memory.get_raw_conversation_history(session_id)
        if not raw_messages:
            return "We haven't discussed any topics yet in this conversation. Feel free to ask me about Arma Reforger modding!"
        
        # Use dynamic AI behavior for memory-only response
        logger.info("💭 Using dynamic AI behavior for memory-only analysis")
        
        try:
            # Create mock empty context docs for the analysis
            mock_docs = []
            
            analysis_result = self.ai_behavior.analyze_and_respond(
                query, mock_docs, conversation_context, response_mode="memory_only"
            )
            
            prompt = analysis_result["prompt"]
            
            return await self.call_ollama_fast(prompt)
            
        except Exception as e:
            logger.error(f"❌ Memory analysis failed: {e}")
            # Fallback to simple memory prompt
            prompt = f"""You are an Arma Reforger assistant. Answer based only on our conversation history.

{conversation_context}

User Question: {query}

Answer based only on what we've actually discussed:"""
            
            return await self.call_ollama_fast(prompt)
    
    async def reform_question_with_ai(self, question: str, session_id: str, memory) -> str:
        """Use enhanced memory context for question reform"""
        
        # ENHANCED: Use enhanced conversation context
        conversation_context = memory.get_enhanced_conversation_context(session_id, question)
        
        logger.info("🤖 Using dynamic AI behavior for analytical question reform")
        
        try:
            # Use dynamic AI behavior for reform
            analysis_result = self.ai_behavior.analyze_and_respond(
                question, [], conversation_context, response_mode="reform"
            )
            
            prompt = analysis_result["prompt"]
            
            return await self.call_ollama_fast(prompt)
            
        except Exception as e:
            logger.error(f"❌ Reform analysis failed: {e}")
            # Fallback to simple reform
            prompt = f"""You are an Arma Reforger expert. Improve this question for better answers.

{conversation_context}

Original: "{question}"

Create 3 improved versions that are more specific and technical.

Answer:"""
            
            return await self.call_ollama_fast(prompt)
    
    async def assess_answer_confidence(self, question: str, answer: str, context_docs: List[Document]) -> float:
        """Ask the LLM to assess its confidence in the answer"""
        try:
            confidence_prompt = f"""Based on the question and your answer, rate your confidence on a scale of 0.0 to 1.0.

Question: {question}

Your Answer: {answer}

Rate your confidence (0.0 = no confidence, 1.0 = completely confident):
- 0.0-0.3: Cannot answer well, need more information
- 0.4-0.6: Partial answer, some uncertainty
- 0.7-0.8: Good answer with minor gaps
- 0.9-1.0: Excellent, comprehensive answer

Respond with ONLY a number between 0.0 and 1.0:"""

            confidence_response = await self.call_ollama(confidence_prompt)
            confidence_str = confidence_response.strip()
            
            confidence_match = re.search(r'(\d+\.?\d*)', confidence_str)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 10.0
                confidence = max(0.0, min(1.0, confidence))
                return confidence
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Error in confidence assessment: {e}")
            return 0.5
    
    # DYNAMIC: Debug Method for Evidence-Based Analysis
    async def debug_dynamic_analysis(self, query: str, context_docs: List[Document], session_id: str, memory) -> Dict[str, Any]:
        """Debug dynamic AI behavior analysis (useful for testing evidence discovery)"""
        
        conversation_context = memory.get_conversation_context(session_id, max_exchanges=3)
        
        try:
            # Run dynamic analysis without generating response
            analysis_result = self.ai_behavior.analyze_and_respond(
                query, context_docs, conversation_context, response_mode="reasoning"
            )
            
            # Safe access to results
            reasoning_result = analysis_result.get("reasoning_result")
            metadata = analysis_result.get("analysis_metadata", {})
            
            debug_info = {
                "should_use_analysis": self.ai_behavior.should_use_intelligent_analysis(query),
                "analysis_mode": analysis_result.get("analysis_mode", "unknown"),
                "confidence": analysis_result.get("analysis_confidence", 0.0),
                "evidence_strength": analysis_result.get("evidence_strength", 0.0),
                "reasoning_quality": analysis_result.get("reasoning_quality", 0.0),
                "prompt_length": len(analysis_result.get("prompt", "")),
                "token_estimate": analysis_result.get("token_estimate", 0),
                "query_intent": metadata.get("query_intent", "unknown"),
                "problem_assumed": metadata.get("problem_assumed", False)
            }
            
            # Add reasoning-specific info if available
            if reasoning_result:
                debug_info.update({
                    "solution_candidates": [
                        {
                            "text": candidate["text"][:100] + "..." if len(candidate["text"]) > 100 else candidate["text"],
                            "type": candidate["type"],
                            "score": candidate["score"],
                            "confidence": candidate["confidence"]
                        }
                        for candidate in reasoning_result.solution_candidates[:3]
                    ],
                    "reasoning_chain": reasoning_result.reasoning_chain,
                    "document_insights": reasoning_result.document_insights
                })
            
            return debug_info
            
        except Exception as e:
            logger.error(f"❌ Debug analysis failed: {e}")
            return {
                "error": str(e),
                "should_use_analysis": False,
                "analysis_mode": "error",
                "confidence": 0.0
            }
    
    # Document Retrieval Methods (unchanged)
    async def retrieve_documents_smart(self, query: str, max_docs: int, preferred_categories: Optional[List[str]], boost_keywords: List[str]) -> List[Document]:
        """Retrieve documents with smart category prioritization"""
        
        # Determine categories to search
        if preferred_categories:
            search_categories = [cat for cat in preferred_categories if cat in self.vector_stores]
            if not search_categories:
                search_categories = list(self.vector_stores.keys())
        else:
            search_categories = list(self.vector_stores.keys())
        
        logger.info(f"🔍 Smart search in categories: {search_categories}")
        
        all_results = []
        
        # Allocation strategy
        if len(search_categories) == 1:
            docs_per_category = max_docs
        else:
            if preferred_categories and set(preferred_categories) == {"Source_Code", "API_Reference"}:
                docs_per_category = max_docs // 2
                logger.info(f"📊 CODE+API split: {docs_per_category} docs per category")
            else:
                docs_per_category = max(10, max_docs // 2)
                logger.info(f"📊 Regular allocation: {docs_per_category} docs per category")
        
        for category in search_categories:
            if category not in self.vector_stores:
                continue
                
            try:
                vector_store = self.vector_stores[category]
                enhanced_query = query
                if boost_keywords:
                    enhanced_query += " " + " ".join(boost_keywords)
                
                try:
                    results = vector_store.similarity_search(enhanced_query, k=docs_per_category + 10)
                    logger.info(f"📄 Retrieved {len(results)} documents from {category}")
                except Exception as e:
                    logger.warning(f"⚠️ Direct search failed for {category}: {e}")
                    retriever = self.retrievers[category]
                    results = retriever.invoke(enhanced_query)
                    logger.info(f"📄 Retriever fallback: {len(results)} documents from {category}")
                
                # Category bonuses
                if category in ["Source_Code", "API_Reference"]:
                    category_limit = min(docs_per_category + 8, len(results))
                elif preferred_categories and category in preferred_categories:
                    category_limit = min(docs_per_category + 3, len(results))
                else:
                    category_limit = min(docs_per_category, len(results))
                
                results = results[:category_limit]
                all_results.extend(results)
                logger.info(f"📄 Using {len(results)} documents from {category}")
                
            except Exception as e:
                logger.error(f"❌ Error retrieving from {category}: {e}")
        
        final_results = all_results[:max_docs]
        logger.info(f"🎯 Final result: {len(final_results)} documents")
        
        return final_results
    
    async def retrieve_documents_quick(self, query: str, max_docs: int, categories: List[str]) -> List[Document]:
        """Quick document retrieval - simplified and fast"""
        all_results = []
        
        for category in categories:
            if category not in self.vector_stores:
                continue
                
            try:
                vector_store = self.vector_stores[category]
                
                # Simple similarity search - no MMR, no fancy allocation
                results = vector_store.similarity_search(query, k=max_docs)
                logger.info(f"⚡ Quick retrieved {len(results)} documents from {category}")
                
                all_results.extend(results)
                break  # Only search first category for speed
                
            except Exception as e:
                logger.error(f"❌ Error in quick retrieval from {category}: {e}")
        
        return all_results[:max_docs]
    
    # System Management Methods (unchanged)
    async def test_ollama_connection(self) -> bool:
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def get_system_stats(self, memory) -> SystemStats:
        """Get system statistics"""
        stats = {
            "categories": {},
            "total_documents": 0,
            "vector_stores": list(self.vector_stores.keys())
        }
        
        for category, vector_store in self.vector_stores.items():
            try:
                collection = vector_store._collection
                count = collection.count()
                
                stats["categories"][category] = {
                    "document_count": count
                }
                stats["total_documents"] += count
                
            except Exception as e:
                logger.error(f"Error getting stats for {category}: {e}")
                stats["categories"][category] = {"error": str(e)}
        
        return SystemStats(
            total_documents=stats["total_documents"],
            categories=stats["categories"],
            vector_stores=stats["vector_stores"],
            system_status="ready" if self.system_ready else "initializing",
            active_sessions=memory.get_active_sessions_count(),
            total_conversations=memory.get_total_conversations_count()
        )
    
    async def health_check(self, memory):
        """System health check"""
        # Test Ollama connection
        ollama_status = await self.test_ollama_connection()
        
        return {
            "status": "healthy" if self.system_ready else "initializing",
            "system_ready": self.system_ready,
            "ollama_connection": ollama_status,
            "vector_stores": list(self.vector_stores.keys()),
            "active_sessions": memory.get_active_sessions_count(),
            "timestamp": datetime.now().isoformat()
        }