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

logger = logging.getLogger(__name__)

class LLMManager:
    """Handles all LLM communication, system initialization, and vector store management"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.embeddings = None
        self.vector_stores = {}
        self.retrievers = {}
        self.system_ready = False
        
        # Initialize system
        self._initialize_system()
    
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
                        "num_ctx": 1024,  # REDUCED from 2048 for memory queries
                        "num_predict": 512  # REDUCED from 1024 for shorter responses
                    }
                },
                timeout=60  # REDUCED from 180 - fail fast for memory queries
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Error: Could not get response from AI model."
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Sorry, the AI is taking too long. Try asking a simpler question or restart the system."
    
    # Answer Generation Methods
    async def generate_answer(self, query: str, context_docs: List[Document], session_id: str, memory) -> str:
        """Generate answer using retrieved context and conversation memory"""
        if not context_docs:
            return "I couldn't find relevant information in the Arma Reforger documentation to answer your question."
        
        # Get conversation context
        conversation_context = memory.get_conversation_context(session_id, max_exchanges=3)
        
        # Prepare RAG context
        context_parts = []
        for i, doc in enumerate(context_docs):
            source_info = f"[{doc.metadata.get('main_category', 'Unknown')}/{doc.metadata.get('sub_category', 'Unknown')}]"
            context_parts.append(f"Document {i+1} {source_info}:\n{doc.page_content}\n")
        
        rag_context = "\n".join(context_parts)
        
        prompt = f"""You are an expert Arma Reforger modding assistant with deep knowledge of Enfusion Script and the Arma Reforger modding ecosystem. You maintain conversation context and can reference previous discussions.

{conversation_context}

Current documentation context:
{rag_context}

Current user question: {query}

Instructions:
- Consider the conversation history when answering
- Reference previous topics when relevant (e.g., "As we discussed earlier...")
- Provide a detailed, helpful answer based on the documentation
- When providing code examples, use proper Enfusion Script syntax
- Format code blocks with proper indentation and structure
- Include specific examples from the documentation when relevant
- If the question involves coding, provide complete, working code examples
- Use proper SCR_ class naming conventions for Arma Reforger
- If multiple approaches exist, mention the alternatives
- Be practical and actionable in your response
- If you're not certain about something, say so
- Structure your response with clear headers when appropriate
- If this continues a previous discussion, acknowledge the context naturally

Answer:"""

        return await self.call_ollama(prompt)
    
    async def generate_answer_quick(self, query: str, context_docs: List[Document], session_id: str, memory) -> str:
        """Generate quick answer with optimized prompt"""
        if not context_docs:
            return "I couldn't find relevant information for this quick query. Try a more detailed question or use a standard prefix."
        
        # Minimal conversation context for speed
        conversation_context = memory.get_conversation_context(session_id, max_exchanges=1)
        
        # Simplified context preparation
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Only use first 3 docs
            context_parts.append(f"Doc {i+1}: {doc.page_content[:300]}...")  # Truncate for speed
        
        rag_context = "\n".join(context_parts)
        
        # Shorter prompt for quick responses
        prompt = f"""You are an Arma Reforger expert. Give a concise, helpful answer.

{conversation_context}

Context:
{rag_context}

Question: {query}

Provide a focused, practical answer. Use Enfusion Script examples if relevant. Keep it concise but useful.

Answer:"""

        return await self.call_ollama_fast(prompt)
    
    async def generate_memory_only_answer(self, query: str, session_id: str, memory) -> str:
        """Generate answer using only conversation memory (no documents) - FIXED HALLUCINATION"""
        conversation_context = memory.get_conversation_context(session_id, max_exchanges=10)  # Get more context
        
        if not conversation_context or "No conversation history available" in conversation_context:
            return "We haven't discussed any topics yet in this conversation. Feel free to ask me about Arma Reforger modding!"
        
        # Get raw messages for additional verification
        raw_messages = memory.get_raw_conversation_history(session_id)
        if not raw_messages:
            return "We haven't discussed any topics yet in this conversation. Feel free to ask me about Arma Reforger modding!"
        
        # ULTRA-STRICT prompt to prevent hallucination
        prompt = f"""You are an Arma Reforger assistant. Answer ONLY about the conversation history provided below.

IMPORTANT: DO NOT MAKE UP OR INVENT ANY CONTENT. Only reference what is explicitly shown.

{conversation_context}

User Question: {query}

STRICT RULES:
- Only use information that is ACTUALLY written in the conversation above
- If something is not in the conversation history, say "We haven't discussed that yet"
- Do NOT invent topics, questions, or discussions about Revit, Enscape, or anything else
- Be factual and only use the exact content provided
- If the conversation is short, acknowledge that
- Only reference Arma Reforger topics that were actually mentioned

Answer based ONLY on the conversation shown above:"""

        return await self.call_ollama_fast(prompt)
    
    async def reform_question_with_ai(self, question: str, session_id: str, memory) -> str:
        """Use AI to intelligently reform/improve questions"""
        
        # Get minimal conversation context for better reform suggestions
        conversation_context = memory.get_conversation_context(session_id, max_exchanges=1)
        
        # Shorter, more focused reform prompt
        reform_prompt = f"""You are an Arma Reforger modding expert. Improve this question to get better answers.

{conversation_context}

Original: "{question}"

Create 3-4 improved versions that are:
- More specific with Arma Reforger terms (SCR_ classes, Enfusion Script)
- Include technical context 
- Actionable for experts

Format:
**Improved Versions:**
1. **[Better question 1]**
   - Why better: [brief reason]

2. **[Better question 2]** 
   - Why better: [brief reason]

3. **[Better question 3]**
   - Why better: [brief reason]

**💡 Key Improvements:** [list main changes]
**🎯 Usage:** Pick the version that matches your goal and ask it!

Focus on practical Arma Reforger modding with proper terminology."""

        return await self.call_ollama_fast(reform_prompt)
    
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
    
    # Document Retrieval Methods
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
    
    # System Management Methods
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