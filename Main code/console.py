import os
import time
import uuid
import sqlite3

class ConsoleChatInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session_id = str(uuid.uuid4())  # Create unique session for console
        
    def start_chat(self):
        """Start the console chat interface"""
        print("🚀 Arma Reforger RAG - Console Chat Interface")
        print("=" * 60)
        print(f"💬 Session ID: {self.session_id[:8]}...")
        
        # Check API health
        self._check_api_health()
        
        print("💡 Tips:")
        print("  • Ask me anything about Arma Reforger modding!")
        print("  • I'll chat naturally by default, but auto-detect when you need docs")
        print("  • Use prefixes for explicit control over search behavior")
        print("  • Type 'help' for commands and all available prefixes")
        print("  • Type 'quit' or 'exit' to quit")
        print("\n🧠 Smart Detection Examples:")
        print("  • 'Hi there!' → 💬 Natural conversation")
        print("  • 'Tell me about weapons' → 🔍 Auto-detects technical content")
        print("  • 'I don't understand bones' → 🔍 Auto-detects need for help")
        print("  • 'Thanks for explaining' → 💬 Stays conversational")
        print("\n🔍 Basic Document Search:")
        print("  • 'base_search <question>' - Explicit search (15 documents)")
        print("  • Example: 'base_search how to create weapons'")
        print("\n⚡ Quick Prefixes (Ultra-Fast):")
        print("  • 'quick_doc' - Documentation (12 docs, ~5-8s)")
        print("  • 'quick_code' - Source_Code (25 docs, ~8-12s)")
        print("  • 'quick_api' - API_Reference (20 docs, ~6-10s)")
        print("  • 'quick_all' - All categories (18 docs, ~10-15s)")
        print("\n🎯 Standard Prefixes (Balanced Performance):")
        print("  • 'standard_doc' - Documentation (30 docs, ~15s)")
        print("  • 'standard_code' - Source_Code (75 docs, ~45s)")
        print("  • 'standard_api' - API_Reference (100 docs, ~60s)")
        print("  • 'standard_code+api' - Both code categories (split allocation)")
        print("  • 'standard_all' - All categories (balanced search)")
        print("\n🚀 Force Prefixes (Maximum Power):")
        print("  • 'force_doc' - All Documentation docs")
        print("  • 'force_code' - All Source_Code docs")
        print("  • 'force_api' - All API_Reference docs")
        print("  • 'force_code+api' - Full split allocation")
        print("  • 'force_all' - Maximum comprehensive search")
        print("  • 'force_benchmark' - Flexible benchmark testing")
        print("\n🧠 Dynamic Prefixes (Adaptive - Progressive with LLM Scoring):")
        print("  • 'dynamic_doc' - Smart expansion (8→30 docs)")
        print("  • 'dynamic_code' - Adaptive code search")
        print("  • 'dynamic_api' - Progressive API search")
        print("  • 'dynamic_code+api' - Intelligent split allocation")
        print("  • 'dynamic_all' - Adaptive comprehensive search")
        print("\n🤖 AI-Powered Question Improvement:")
        print("  • 'reform [your question]' - AI analyzes and improves your question")
        print("  • Fast processing (optimized LLM settings, no document search)")
        print("  • Example: 'reform how to make weapon' → Multiple improved versions")
        print("\n💬 Memory Commands:")
        print("  • 'memory <question>' - Ask about our conversation history")
        print("  • 'history' - Show conversation history")
        print("  • 'clear' - Clear conversation memory")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                elif user_input.lower() == 'health':
                    self._check_api_health()
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_conversation_history()
                    continue
                
                elif user_input.lower().startswith('memory '):
                    # Memory-only query - skip document search
                    memory_question = user_input[7:].strip()  # Remove "memory "
                    if memory_question:
                        self._process_memory_query(memory_question)
                    else:
                        print("💭 Please ask a memory-related question after 'memory'")
                        print("💡 Example: memory what was the first topic we covered?")
                    continue
                
                elif user_input.lower() == 'clear':
                    self._clear_conversation()
                    continue
                
                elif user_input.lower() == 'debug':
                    self._show_debug_info()
                    continue
                
                # Process the query
                self._process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _check_api_health(self):
        """Check API health"""
        try:
            import requests
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Status: {health_data['status']}")
                print(f"🗃️ Vector Stores: {', '.join(health_data['vector_stores'])}")
                print(f"💬 Active Sessions: {health_data.get('active_sessions', 0)}")
                if not health_data['ollama_connection']:
                    print("⚠️ Warning: Ollama connection failed")
            else:
                print(f"❌ API Health Check Failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print("💡 Make sure the API server is running!")
            print("💡 Waiting a bit longer for API to start...")
            time.sleep(2)
    
    def _show_stats(self):
        """Show system statistics"""
        try:
            import requests
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                print("\n📊 SYSTEM STATISTICS:")
                print(f"📄 Total Documents: {stats['total_documents']}")
                print(f"💬 Active Sessions: {stats['active_sessions']}")
                print(f"📝 Total Conversations: {stats['total_conversations']}")
                print("📂 Categories:")
                for category, info in stats['categories'].items():
                    print(f"   {category}: {info['document_count']} documents")
            else:
                print(f"❌ Failed to get stats: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def _show_conversation_history(self):
        """Show conversation history for current session"""
        try:
            import requests
            response = requests.get(f"{self.api_url}/conversation/{self.session_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                summary = data['summary']
                messages = data['messages']
                
                print(f"\n💬 CONVERSATION HISTORY (Session: {self.session_id[:8]}...):")
                print(f"📊 Length: {summary['length']} exchanges")
                print(f"🏷️ Topics: {', '.join(summary['topics']) if summary['topics'] else 'None detected'}")
                
                if messages:
                    print("\n📝 Recent Messages:")
                    for i, msg in enumerate(messages[-10:], 1):  # Show last 10 messages
                        role_emoji = "🤔" if msg['role'] == 'user' else "🤖"
                        timestamp = msg['timestamp'][:19].replace('T', ' ')
                        print(f"  {i:2d}. {role_emoji} {msg['role'].title()} [{timestamp}]:")
                        print(f"      {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                else:
                    print("📝 No messages in this session yet.")
            else:
                print(f"❌ Failed to get conversation history: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
    
    def _show_debug_info(self):
        """Show debug information about database and sessions"""
        print("\n🔧 DEBUG INFORMATION:")
        print(f"💬 Session ID: {self.session_id}")
        print(f"🌐 API URL: {self.api_url}")
        
        # Check if database file exists
        db_path = r"C:\ArmaModdingRAG\conversations.db"
        if os.path.exists(db_path):
            print(f"✅ Database file exists: {db_path}")
            try:
                import sqlite3
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                    count = cursor.fetchone()[0]
                    print(f"📊 Total messages in database: {count}")
                    
                    cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
                    sessions = cursor.fetchone()[0]
                    print(f"💬 Total sessions in database: {sessions}")
                    
                    if count > 0:
                        cursor = conn.execute("SELECT session_id, role, timestamp FROM conversations ORDER BY timestamp DESC LIMIT 5")
                        recent = cursor.fetchall()
                        print("📝 Recent database entries:")
                        for session_id, role, timestamp in recent:
                            print(f"   {session_id[:8]}... | {role} | {timestamp[:19]}")
            except Exception as e:
                print(f"❌ Database query failed: {e}")
        else:
            print(f"❌ Database file not found: {db_path}")
            print(f"📁 Directory exists: {os.path.exists(os.path.dirname(db_path))}")
    
    def _clear_conversation(self):
        """Clear conversation memory"""
        try:
            import requests
            response = requests.delete(f"{self.api_url}/conversation/{self.session_id}/clear", timeout=10)
            if response.status_code == 200:
                print("✅ Conversation memory cleared!")
                print("🔄 Starting fresh conversation...")
            else:
                print(f"❌ Failed to clear conversation: {response.status_code}")
        except Exception as e:
            print(f"❌ Error clearing conversation: {e}")
    
    def _process_memory_query(self, question: str):
        """Process a memory-only query without document search"""
        print(f"\n💭 Memory Query: {question}")
        print("🧠 Searching conversation history only (no documents)...")
        
        try:
            import requests
            
            # Create special payload for memory-only queries
            payload = {
                "message": f"MEMORY_ONLY: {question}",
                "session_id": self.session_id,
                "max_docs": 0,  # Skip document retrieval
                "use_smart_processing": False  # Use basic processing for memory queries
            }
            
            start_time = time.time()
            
            response = requests.post(f"{self.api_url}/chat", json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                self._display_memory_response(result)
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                        
        except Exception as e:
            print(f"❌ Error processing memory query: {e}")
    
    def _display_memory_response(self, result: dict):
        """Display response for memory-only queries"""
        print("\n" + "=" * 60)
        print("🧠 MEMORY RESPONSE:")
        print("=" * 60)
        print(result['answer'])
        
        print(f"\n💭 Memory-only query | No documents searched")
        print(f"💬 Session: {result['session_id'][:8]}... | Conversation length: {result['conversation_length']} exchanges")
        print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        print("=" * 60)
    
    def _process_query(self, question: str):
        """Process a user query"""
        
        # Check if this is a reform query - show appropriate message
        if question.strip().lower().startswith('reform '):
            print("\n🤖 AI-powered question improvement...")
        elif any(question.strip().lower().startswith(prefix) for prefix in ['quick_', 'standard_', 'force_', 'dynamic_', 'base_search']):
            print("\n🔍 Searching documentation...")
        else:
            print("\n🤔 Analyzing query...")
        
        try:
            import requests
            
            # Prepare request with session ID
            payload = {
                "message": question,
                "session_id": self.session_id,
                "use_smart_processing": True  # Always enable smart processing
            }
            
            start_time = time.time()
            
            response = requests.post(f"{self.api_url}/chat", json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                self._display_response(result)
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                        
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    def _display_response(self, result: dict):
        """Display the API response"""
        
        # Special handling for AI reform responses
        if result.get('query_type') == 'ai_reform':
            print("\n" + "=" * 60)
            print("🤖 AI-POWERED QUESTION REFORM:")
            print("=" * 60)
            print(result['answer'])
            
            print(f"\n🤖 AI Reform | No documents searched (0 sources)")
            print(f"💬 Session: {result['session_id'][:8]}... | Conversation length: {result['conversation_length']} exchanges")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            if result.get('suggested_follow_ups'):
                print(f"\n💡 Next Steps:")
                for i, suggestion in enumerate(result['suggested_follow_ups'], 1):
                    print(f"   {i}. {suggestion}")
            print("=" * 60)
            return
        
        # Memory-only responses
        if result.get('query_type') == 'memory_query':
            print("\n" + "=" * 60)
            print("💬 CONVERSATION RESPONSE:")
            print("=" * 60)
            print(result['answer'])
            
            print(f"\n💬 Conversation mode | No documents searched")
            print(f"💬 Session: {result['session_id'][:8]}... | Conversation length: {result['conversation_length']} exchanges")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            if result.get('suggested_follow_ups'):
                print(f"\n💡 Follow-up suggestions:")
                for i, suggestion in enumerate(result['suggested_follow_ups'], 1):
                    print(f"   {i}. {suggestion}")
            print("=" * 60)
            return
        
        # Auto-search responses (show why it chose to search)
        if result.get('query_type') == 'regular_query' and not result.get('category_prefix_used'):
            print("\n" + "=" * 60)
            print("🔍 AUTO-SEARCH RESPONSE:")
            print("=" * 60)
            print(result['answer'])
            
            print(f"\n🔍 Auto-detected need for document search | {result['context_used']} documents used")
            # Show reasoning if available in logs
            print(f"💡 Why: Smart detection found technical content")
        else:
            # Regular response handling (with explicit prefix)
            print("\n" + "=" * 60)
            print("📝 ANSWER:")
            print("=" * 60)
            print(result['answer'])
        
        if result.get('sources'):
            print(f"\n📚 Sources ({result['context_used']} documents used):")
            
            # Show only first 15 sources, but indicate total count
            sources_to_show = result['sources'][:15]
            for i, source in enumerate(sources_to_show, 1):
                print(f"  {i:2d}. {source['filename']} ({source['category']})")
            
            # If there are more sources, show summary
            if len(result['sources']) > 15:
                remaining = len(result['sources']) - 15
                print(f"  ... and {remaining} more documents")
                print(f"      (showing first 15 of {len(result['sources'])} total sources)")

        
        # Show smart processing info if available
        if 'query_type' in result:
            print(f"\n🧠 Smart Processing:")
            print(f"   Query Type: {result['query_type']}")
            print(f"   Confidence: {result.get('confidence_score', 0):.2f}")
            
            # Show progressive retrieval info
            if result['context_used'] > 0:
                efficiency = (result['context_used'] / 50) * 100  # Assume max 50 for calculation
                print(f"   Documents Used: {result['context_used']} ({efficiency:.1f}% of max)")
            
            if result.get('intent_detected') and result.get('intent_detected') != 'direct_question':
                print(f"   Intent: {result['intent_detected']}")
            
            if result.get('context_enhanced'):
                print(f"   Context Enhanced: Yes")
            
            if result.get('category_prefix_used'):
                # Updated prefix type detection
                if result['category_prefix_used'] == 'reform':
                    prefix_type = "🤖 AI-Reform"
                elif result['category_prefix_used'] == 'base_search':
                    prefix_type = "🔍 Base-Search"
                elif result['category_prefix_used'].startswith('dynamic_'):
                    prefix_type = "🧠 Dynamic"
                elif result['category_prefix_used'].startswith('force_'):
                    prefix_type = "🚀 Force"
                elif result['category_prefix_used'].startswith('quick_'):
                    prefix_type = "⚡ Quick"
                else:
                    prefix_type = "🎯 Standard"
                print(f"   Prefix Used: {prefix_type} '{result['category_prefix_used']}'")
            
            if result.get('corrections_made'):
                print(f"   Corrections: {', '.join(result['corrections_made'])}")
            
            if result.get('multi_part'):
                print(f"   Multi-part Query: Yes")
            
            if result.get('validation_issues'):
                print(f"   Validation: {len(result['validation_issues'])} issues noted")
            
            if result.get('suggested_follow_ups'):
                print(f"\n💡 Suggested Follow-ups:")
                for i, suggestion in enumerate(result['suggested_follow_ups'], 1):
                    print(f"   {i}. {suggestion}")
        
        print(f"\n💬 Session: {result['session_id'][:8]}... | Conversation length: {result['conversation_length']} exchanges")
        print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        print("=" * 60)
    
    def _show_help(self):
        """Show help information"""
        print("\n💡 AVAILABLE COMMANDS:")
        print("  help    - Show this help message")
        print("  stats   - Show system statistics") 
        print("  health  - Check API health")
        print("  history - Show conversation history")
        print("  clear   - Clear conversation memory")
        print("  debug   - Show database debug info")
        print("  quit/exit - Exit the chat")
        print("\n💬 CONVERSATION MODE (Default):")
        print("  • No prefix = Natural conversation (no document search)")
        print("  • 'Hi, what can you help me with?' - Just chat")
        print("  • 'Can you explain that differently?' - Uses conversation memory")
        print("  • 'Thanks for the help!' - Natural responses")
        print("\n🔍 DOCUMENT SEARCH PREFIXES:")
        print("  🔍 BASIC SEARCH:")
        print("    • base_search <question> - Search docs (15 docs, balanced)")
        print("    • Example: 'base_search weapon creation workflow'")
        print("  ⚡ QUICK (Ultra-Fast):")
        print("    • quick_doc <question> - Documentation (12 docs, ~5-8s)")
        print("    • quick_code <question> - Source_Code (25 docs, ~8-12s)")
        print("    • quick_api <question> - API_Reference (20 docs, ~6-10s)")
        print("    • quick_all <question> - All categories (18 docs, ~10-15s)")
        print("  🎯 STANDARD (Balanced):")
        print("    • standard_doc <question> - Documentation (30 docs, ~15s)")
        print("    • standard_code <question> - Source_Code (75 docs, ~45s)")
        print("    • standard_api <question> - API_Reference (100 docs, ~60s)")
        print("  🚀 FORCE & 🧠 DYNAMIC available for comprehensive analysis")
        print("  🤖 SPECIAL:")
        print("    • reform <question> - AI-powered question improvement")
        print("\n📝 EXAMPLE USAGE:")
        print("  • 'Hello!' → Natural chat")
        print("  • 'base_search how to import models' → Basic doc search")
        print("  • 'quick_code weapon damage function' → Fast code search")
        print("  • 'reform my script is broken' → AI question improvement")
        print("\n💬 MEMORY FEATURES:")
        print("  • I remember our conversation within this session")
        print("  • You can refer to previous topics: 'Can you expand on that?'")
        print("  • I maintain context across multiple questions")
        print("  • Use 'clear' to start a fresh conversation")
        print("  • Use 'memory <question>' to ask about our conversation only")
        print("\n💭 MEMORY-ONLY EXAMPLES:")
        print("  • memory what topics have we discussed?")
        print("  • memory what was my first question?")
        print("  • memory can you summarize our conversation?")
        print("  • memory what examples did you show me?")