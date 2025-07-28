"""
Arma Reforger RAG Assistant - Main WebUI Application
Simplified version with text export and hidden import feature
"""

import streamlit as st
import requests
import json
import time
import uuid
import re
from datetime import datetime
from typing import Dict, List, Any

# Page config
st.set_page_config(
    page_title="Arma Reforger RAG Assistant",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS with balanced typography for tutorial sections
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    button[kind="header"] {
        display: none !important;
    }
    
    /* Force sidebar to always be visible and prevent collapse */
    .css-1d391kg {
        min-width: 300px !important;
    }
    
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        min-width: 300px !important;
        width: 300px !important;
    }
    
    /* Hide the sidebar close button */
    button[kind="header"][data-testid="collapsedControl"] {
        display: none !important;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        background-color: #fafafa;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        background-color: white;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        border-left: 4px solid #4caf50;
        margin-right: 2rem;
    }
    
    .assistant-message p, .assistant-message div {
        border-left: none !important;
    }
    
    .thinking-section {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        font-family: monospace;
    }
    
    .processing-info {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .source-item {
        background-color: #ffffff;
        border-left: 4px solid #1e3c72;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.85rem;
    }
    
    .tutorial-section {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%);
        border: 2px solid #4a90e2;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.15);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .tutorial-section h3 {
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
        color: #2c3e50;
    }
    
    .tutorial-section h4 {
        font-size: 1.1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        color: #34495e;
    }
    
    .tutorial-section p, .tutorial-section li {
        font-size: 1rem !important;
        line-height: 1.6 !important;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .tutorial-section ul, .tutorial-section ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .tutorial-section code {
        background-color: rgba(27, 31, 35, 0.05);
        border-radius: 3px;
        font-size: 0.95rem !important;
        padding: 2px 4px;
        color: #d73a49;
        font-weight: 600;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class ArmaRAGWebUI:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = None
        if 'show_reasoning' not in st.session_state:
            st.session_state.show_reasoning = {}
        if 'layout_mode' not in st.session_state:
            st.session_state.layout_mode = "Standard"
    
    def check_api_health(self) -> Dict:
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "error", "message": f"API returned {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_system_stats(self) -> Dict:
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def send_message(self, message: str, use_smart_processing: bool = True) -> Dict:
        try:
            payload = {
                "message": message,
                "session_id": st.session_state.session_id,
                "use_smart_processing": use_smart_processing
            }
            response = requests.post(f"{self.api_url}/chat", json=payload, timeout=300)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def clear_conversation(self):
        try:
            response = requests.delete(f"{self.api_url}/conversation/{st.session_state.session_id}/clear", timeout=10)
            if response.status_code == 200:
                st.session_state.conversation_history = []
                st.session_state.show_reasoning = {}
                return True
        except:
            pass
        return False
    
    def export_chat_as_text(self) -> str:
        """Export chat history as formatted plain text"""
        chat_text = f"Arma Reforger RAG Assistant - Chat Export\n"
        chat_text += f"=" * 50 + "\n"
        chat_text += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        chat_text += f"Session ID: {st.session_state.session_id}\n"
        chat_text += f"Total Messages: {len(st.session_state.conversation_history)}\n"
        chat_text += "=" * 50 + "\n\n"
        
        for i, message in enumerate(st.session_state.conversation_history, 1):
            role = "👤 USER" if message["role"] == "user" else "🤖 ASSISTANT"
            timestamp = message.get("timestamp", "")
            
            chat_text += f"[{i}] {role} ({timestamp})\n"
            chat_text += "-" * 40 + "\n"
            
            content = message["content"]
            if message["role"] == "assistant":
                # Remove thinking tags for text export
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            chat_text += f"{content}\n"
            
            # Add processing info if available
            if "processing_info" in message:
                info = message["processing_info"]
                chat_text += f"\n📊 PROCESSING INFO:\n"
                chat_text += f"   • Processing Time: {info.get('processing_time', 0):.2f} seconds\n"
                chat_text += f"   • Documents Used: {info.get('context_used', 0)}\n"
                chat_text += f"   • Confidence Score: {info.get('confidence_score', 0):.2f}\n"
                chat_text += f"   • Query Type: {info.get('query_type', 'Unknown')}\n"
                if info.get('category_prefix_used'):
                    chat_text += f"   • Prefix Used: {info.get('category_prefix_used')}\n"
                
                # Add sources if available
                sources = info.get('sources', [])
                if sources:
                    chat_text += f"   • Sources ({len(sources)} documents):\n"
                    for j, source in enumerate(sources[:5], 1):  # Show first 5 sources
                        chat_text += f"     {j}. {source.get('filename', 'Unknown')} ({source.get('category', 'Unknown')})\n"
                    if len(sources) > 5:
                        chat_text += f"     ... and {len(sources) - 5} more sources\n"
            
            chat_text += "\n" + "=" * 50 + "\n\n"
        
        return chat_text
    
    def extract_thinking_content(self, text: str) -> tuple:
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, text, re.DOTALL)
        if matches:
            thinking_content = matches[0].strip()
            clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
            return clean_text, thinking_content
        return text, None
    
    def render_code_block(self, code: str, language: str = ""):
        st.code(code, language=language)
    
    def format_message_content(self, content: str):
        parts = re.split(r'```(\w*)\n(.*?)\n```', content, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if i % 3 == 0:
                if part.strip():
                    st.markdown(part)
            elif i % 3 == 1:
                continue
            else:
                language = parts[i-1] if i > 0 else ""
                self.render_code_block(part, language)
    
    def render_chat_message(self, message: Dict, index: int):
        is_user = message["role"] == "user"
        css_class = "user-message" if is_user else "assistant-message"
        
        with st.container():
            st.markdown(f'<div class="chat-message {css_class}">', unsafe_allow_html=True)
            
            role_emoji = "🤔" if is_user else "🤖"
            role_name = "You" if is_user else "Assistant"
            timestamp = message.get("timestamp", "")
            
            st.markdown(f"**{role_emoji} {role_name}** *{timestamp}*")
            
            if is_user:
                st.markdown(message["content"])
            else:
                content = message["content"]
                clean_content, thinking_content = self.extract_thinking_content(content)
                self.format_message_content(clean_content)
                
                if thinking_content:
                    thinking_key = f"thinking_{index}"
                    if thinking_key not in st.session_state.show_reasoning:
                        st.session_state.show_reasoning[thinking_key] = False
                    
                    if st.button(f"{'🧠 Hide Reasoning' if st.session_state.show_reasoning[thinking_key] else '🧠 Show Reasoning'}", 
                               key=f"toggle_{thinking_key}"):
                        st.session_state.show_reasoning[thinking_key] = not st.session_state.show_reasoning[thinking_key]
                    
                    if st.session_state.show_reasoning[thinking_key]:
                        st.markdown('<div class="thinking-section">', unsafe_allow_html=True)
                        st.markdown("**🧠 AI Reasoning Process:**")
                        st.text(thinking_content)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                if "processing_info" in message:
                    self.render_processing_info(message["processing_info"])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_processing_info(self, info: Dict):
        st.markdown('<div class="processing-info">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⏱️ Processing Time", f"{info.get('processing_time', 0):.2f}s")
        with col2:
            st.metric("📄 Documents Used", info.get('context_used', 0))
        with col3:
            st.metric("🎯 Confidence", f"{info.get('confidence_score', 0):.2f}")
        
        if info.get('query_type'):
            st.text(f"Query Type: {info['query_type']}")
        if info.get('category_prefix_used'):
            st.text(f"Prefix Used: {info['category_prefix_used']}")
        
        if info.get('sources'):
            with st.expander(f"📚 View Sources ({len(info['sources'])} documents)", expanded=False):
                for i, source in enumerate(info['sources'][:10], 1):
                    st.markdown(f'<div class="source-item">', unsafe_allow_html=True)
                    st.markdown(f"**{i}.** {source.get('filename', 'Unknown')}")
                    st.markdown(f"*Category: {source.get('category', 'Unknown')}*")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if len(info['sources']) > 10:
                    st.text(f"... and {len(info['sources']) - 10} more sources")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown('<div class="main-header"><h2>🚁 Arma RAG Control Panel</h2></div>', unsafe_allow_html=True)
            
            st.subheader("🔧 System Status")
            health = self.check_api_health()
            
            if health["status"] == "healthy":
                st.success("✅ API Connected")
                if "data" in health:
                    st.text(f"Vector Stores: {len(health['data'].get('vector_stores', []))}")
            else:
                st.error(f"❌ API Error: {health['message']}")
            
            st.divider()
            
            st.subheader("⚡ Quick Prefixes")
            
            prefix_categories = {
                "⚡ Quick": ["quick_doc", "quick_code", "quick_api", "quick_all"],
                "🎯 Standard": ["standard_doc", "standard_code", "standard_api", "standard_all"],
                "🚀 Force": ["force_doc", "force_code", "force_api", "force_all"],
                "🧠 Dynamic": ["dynamic_doc", "dynamic_code", "dynamic_api", "dynamic_all"],
                "🤖 Special": ["reform", "memory", "base_search"]
            }
            
            for category, prefixes in prefix_categories.items():
                with st.expander(category, expanded=False):
                    for prefix in prefixes:
                        if st.button(prefix, key=f"prefix_{prefix}", use_container_width=True):
                            st.session_state.selected_prefix = prefix
            
            st.divider()
            
            st.subheader("💾 Memory Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Memory", use_container_width=True):
                    if self.clear_conversation():
                        st.success("Memory cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear memory")
            
            with col2:
                if st.button("📄 Export Chat", use_container_width=True):
                    if st.session_state.conversation_history:
                        try:
                            chat_text = self.export_chat_as_text()
                            st.download_button(
                                label="📥 Download TXT",
                                data=chat_text,
                                file_name=f"arma_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
                    else:
                        st.warning("No conversation to export")
            
            # Import feature temporarily hidden
            # TODO: Re-enable when file upload issues are resolved
            
            st.divider()
            
            # Auto-load stats on first render
            if st.session_state.system_stats is None:
                st.session_state.system_stats = self.get_system_stats()
            
            if st.session_state.system_stats:
                st.subheader("📊 System Statistics")
                stats = st.session_state.system_stats
                st.metric("Total Documents", stats.get('total_documents', 0))
                st.metric("Active Sessions", stats.get('active_sessions', 0))
    
    def render_main_chat(self):
        st.markdown('<div class="main-header"><h1>🚁 Arma Reforger RAG Assistant</h1><p>AI-powered modding assistance with smart document search</p></div>', unsafe_allow_html=True)
        
        chat_height = "70vh" if st.session_state.layout_mode == "Full Height" else "50vh"
        st.markdown(f'<div class="chat-container" style="height: {chat_height};">', unsafe_allow_html=True)
        
        if st.session_state.conversation_history:
            for i, message in enumerate(st.session_state.conversation_history):
                self.render_chat_message(message, i)
        else:
            st.markdown("### 👋 Welcome to Arma Reforger RAG Assistant!")
            st.markdown("""
            Ask me anything about Arma Reforger modding! I have access to comprehensive documentation covering:
            
            - **🔧 Modding Tutorials** - Step-by-step guides and workflows
            - **💻 Source Code** - Real examples and implementations from the game
            - **📖 API Reference** - Complete class documentation and method details
            """)
            
            # Quick Start Examples in a styled box
            st.markdown("""
            <div class="tutorial-section">
            <h4>🚀 Quick Start Examples:</h4>
            <ul>
                <li><code>quick_doc what is Enfusion Script?</code> - Fast introduction to scripting basics</li>
                <li><code>standard_code weapon creation example</code> - Balanced code examples for weapons</li>
                <li><code>force_api SCR_WeaponComponent documentation</code> - Complete API reference</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 💬 Ask a Question")
        
        # Show prefix info and clear button (outside form)
        if hasattr(st.session_state, 'selected_prefix'):
            prefix = st.session_state.selected_prefix
            
            # Detailed prefix information
            prefix_info = {
                # Quick Prefixes
                "quick_doc": {
                    "name": "Quick Documentation",
                    "docs": "12 Documentation files",
                    "time": "~5-8 seconds",
                    "description": "Fast answers from tutorial guides and modding documentation",
                    "color": "info"
                },
                "quick_code": {
                    "name": "Quick Code Examples", 
                    "docs": "25 Source_Code files",
                    "time": "~8-12 seconds",
                    "description": "Rapid code examples and implementation patterns",
                    "color": "info"
                },
                "quick_api": {
                    "name": "Quick API Reference",
                    "docs": "20 API_Reference files", 
                    "time": "~6-10 seconds",
                    "description": "Fast class and method documentation lookup",
                    "color": "info"
                },
                "quick_all": {
                    "name": "Quick All Categories",
                    "docs": "18 files (balanced across all categories)",
                    "time": "~10-15 seconds", 
                    "description": "Balanced fast search across Documentation, Code, and API",
                    "color": "info"
                },
                
                # Standard Prefixes
                "standard_doc": {
                    "name": "Standard Documentation",
                    "docs": "30 Documentation files",
                    "time": "~15 seconds",
                    "description": "Sweet spot coverage for tutorials and modding guides",
                    "color": "success"
                },
                "standard_code": {
                    "name": "Standard Code Examples",
                    "docs": "75 Source_Code files", 
                    "time": "~45 seconds",
                    "description": "Pattern-focused examples with comprehensive code coverage",
                    "color": "success"
                },
                "standard_api": {
                    "name": "Standard API Reference",
                    "docs": "100 API_Reference files",
                    "time": "~60 seconds",
                    "description": "Comprehensive class documentation and method details",
                    "color": "success"
                },
                "standard_all": {
                    "name": "Standard All Categories", 
                    "docs": "60 files (balanced across all categories)",
                    "time": "~30-45 seconds",
                    "description": "Increased coverage across Documentation, Code, and API",
                    "color": "success"
                },
                
                # Force Prefixes
                "force_doc": {
                    "name": "Force Documentation",
                    "docs": "50 Documentation files",
                    "time": "~30-60 seconds",
                    "description": "Full coverage of modding documentation without noise",
                    "color": "warning"
                },
                "force_code": {
                    "name": "Force Code Examples",
                    "docs": "200 Source_Code files",
                    "time": "~60-90 seconds", 
                    "description": "Comprehensive but focused code examples and patterns",
                    "color": "warning"
                },
                "force_api": {
                    "name": "Force API Reference",
                    "docs": "300 API_Reference files",
                    "time": "~90-120 seconds",
                    "description": "Complete API documentation with full class coverage",
                    "color": "warning"
                },
                "force_all": {
                    "name": "Force All Categories",
                    "docs": "100 files (comprehensive across all categories)",
                    "time": "~60-120 seconds",
                    "description": "Maximum quality search with doubled coverage",
                    "color": "warning"
                },
                
                # Dynamic Prefixes
                "dynamic_doc": {
                    "name": "Dynamic Documentation",
                    "docs": "8→30 files (progressive expansion)",
                    "time": "~30-120 seconds",
                    "description": "Starts small, expands based on AI confidence (≥0.75 threshold)",
                    "color": "error"
                },
                "dynamic_code": {
                    "name": "Dynamic Code Examples",
                    "docs": "8→100 files (progressive expansion)", 
                    "time": "~30-180 seconds",
                    "description": "Adaptive code search that grows until confident answer found",
                    "color": "error"
                },
                "dynamic_api": {
                    "name": "Dynamic API Reference",
                    "docs": "8→150 files (progressive expansion)",
                    "time": "~30-200 seconds",
                    "description": "Intelligent API expansion based on answer confidence",
                    "color": "error"
                },
                "dynamic_all": {
                    "name": "Dynamic All Categories",
                    "docs": "8→75 files (balanced progressive expansion)",
                    "time": "~30-180 seconds",
                    "description": "Adaptive search across all categories with optimal efficiency",
                    "color": "error"
                },
                
                # Special Prefixes  
                "reform": {
                    "name": "AI Question Reform",
                    "docs": "No document search",
                    "time": "~15-20 seconds",
                    "description": "AI analyzes your question and suggests 3-4 improved versions",
                    "color": "secondary"
                },
                "memory": {
                    "name": "Conversation Memory",
                    "docs": "Chat history analysis",
                    "time": "~10-15 seconds",
                    "description": "Ask about our conversation history and previous topics discussed",
                    "color": "secondary"
                },
                "base_search": {
                    "name": "Base Search",
                    "docs": "Default system behavior",
                    "time": "Variable",
                    "description": "Standard search without prefix optimizations",
                    "color": "secondary"
                }
            }
            
            info = prefix_info.get(prefix, {
                "name": "Unknown Prefix",
                "docs": "Unknown",
                "time": "Unknown", 
                "description": "No information available",
                "color": "secondary"
            })
            
            # Display detailed prefix information with clear button
            col1, col2 = st.columns([4, 1])
            with col1:
                st.info(f"""
                💡 **Selected: {info['name']}** (`{prefix}`)
                
                📄 **Search Scope:** {info['docs']}  
                ⏱️ **Expected Time:** {info['time']}  
                🎯 **Purpose:** {info['description']}
                
                Type your question and it will be automatically prefixed!
                """)
            with col2:
                if st.button("❌ Clear Prefix", use_container_width=True):
                    del st.session_state.selected_prefix
        
        # Input section
        user_input = st.text_area(
            "Your question:",
            height=100,
            placeholder="Ask about Arma Reforger modding... (e.g., 'quick_doc how to create weapons')",
            key="user_input_field",
            value="" if st.session_state.get('user_input_clear', False) else st.session_state.get('user_input_field', "")
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            send_button = st.button("🚀 Send Message", type="primary", use_container_width=True)
        with col2:
            smart_processing = st.checkbox("🧠 Smart Processing", value=True, help="AI analyzes your question, fixes terminology, and optimizes search strategy automatically")
        
        # Smart Processing Explanation
        if smart_processing:
            st.info("💡 **Smart Processing ON:** AI will analyze your question, correct Arma-specific terminology, detect question type, and choose optimal search strategy automatically.")
        else:
            st.warning("⚠️ **Smart Processing OFF:** Direct search without AI optimization. May miss relevant results or use suboptimal search patterns.")
        
        if send_button and user_input and user_input.strip():
            if hasattr(st.session_state, 'selected_prefix'):
                message = f"{st.session_state.selected_prefix} {user_input}"
                del st.session_state.selected_prefix
            else:
                message = user_input
            
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.conversation_history.append(user_message)
            
            with st.spinner("🤔 Processing your question..."):
                result = self.send_message(message, smart_processing)
            
            if result["status"] == "success":
                response_data = result["data"]
                assistant_message = {
                    "role": "assistant",
                    "content": response_data["answer"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_info": {
                        "processing_time": response_data.get("processing_time", 0),
                        "context_used": response_data.get("context_used", 0),
                        "confidence_score": response_data.get("confidence_score", 0),
                        "query_type": response_data.get("query_type", ""),
                        "category_prefix_used": response_data.get("category_prefix_used", ""),
                        "sources": response_data.get("sources", [])
                    }
                }
                st.session_state.conversation_history.append(assistant_message)
                
                # Clear the input and rerun ONLY after successful message processing
                st.session_state.user_input_clear = True
                st.rerun()
            else:
                st.error(f"❌ Error: {result['message']}")
        
        # Clear input after sending
        if st.session_state.get('user_input_clear', False):
            st.session_state.user_input_clear = False
    
    def render_tutorial_page(self):
        st.markdown("# 📚 Arma Reforger RAG Assistant Guide")
        st.markdown("*Complete tutorial and reference for getting the most out of your AI modding assistant*")
        
        st.divider()
        
        # Getting Started
        st.markdown("## 🚀 Getting Started")
        
        st.markdown("""
        ### 📖 What is this system?
        
        The **Arma Reforger RAG Assistant** is an AI-powered modding helper with access to:
        
        - **📚 62,517 documents** across 3 specialized categories
        - **🧠 Smart query processing** that auto-detects your needs
        - **⚡ Four-tier search system** for different speed/depth requirements
        - **💬 Conversation memory** that remembers context across questions
        """)
        
        # Basic Tips
        st.markdown("## 💡 Basic Tips")
        
        st.markdown("""
        ### 🎯 How to ask effective questions
        
        **✅ Good Questions:**
        - `quick_doc how to import weapon models`
        - `standard_code vehicle damage system`
        - `force_api SCR_WeaponComponent documentation`
        
        **💡 Pro Tips:**
        - Be specific about what you want to achieve
        - Mention relevant tools (Blender, Workbench, etc.)
        - Use proper Arma Reforger terminology when known
        """)
        
        # Smart Processing
        st.markdown("## 🧠 Smart Processing System")
        
        st.markdown("""
        ### 🔍 What Smart Processing Does
        
        When the **🧠 Smart Processing** checkbox is enabled, the AI automatically:
        
        - **Analyzes Question Type** - Detects if you want how-to guides, code examples, API docs, or troubleshooting
        - **Fixes Terminology** - Corrects common mistakes like "weaponmanager" → "WeaponManager"
        - **Optimizes Search Strategy** - Chooses the best document categories and search approach
        - **Splits Complex Questions** - Breaks down multi-part queries into manageable pieces
        - **Suggests Improvements** - Offers better ways to phrase your question for optimal results
        
        **💡 Recommendation:** Keep Smart Processing ON for best results. Only disable for very specific direct searches.
        """)
        
        # Prefixes Overview
        st.markdown("## ⚡ Prefix System Overview")
        
        st.markdown("### 🚀 Quick Prefixes (Ultra-fast)")
        st.markdown("""
        - `quick_doc` - Documentation (12 docs, ~5-8s)
        - `quick_code` - Source_Code (25 docs, ~8-12s)
        - `quick_api` - API_Reference (20 docs, ~6-10s)
        - `quick_all` - All categories (18 docs, ~10-15s)
        """)
        
        st.markdown("### 🎯 Standard Prefixes (Balanced)")
        st.markdown("""
        - `standard_doc` - Documentation (30 docs, ~15s)
        - `standard_code` - Source_Code (75 docs, ~45s)
        - `standard_api` - API_Reference (100 docs, ~60s)
        - `standard_all` - All categories (60 docs, ~30-45s)
        """)
        
        st.markdown("### 🚀 Force Prefixes (Maximum power)")
        st.markdown("""
        - `force_doc` - Documentation (50 docs, ~30-60s)
        - `force_code` - Source_Code (200 docs, ~60-90s)
        - `force_api` - API_Reference (300 docs, ~90-120s)
        - `force_all` - All categories (100 docs, ~60-120s)
        """)
        
        st.markdown("### 🧠 Dynamic Prefixes (Adaptive)")
        st.markdown("""
        - `dynamic_doc` - 8→30 docs (progressive expansion)
        - `dynamic_code` - 8→100 docs (progressive expansion)
        - `dynamic_api` - 8→150 docs (progressive expansion)
        - `dynamic_all` - 8→75 docs (balanced progressive)
        """)
        
        st.markdown("### 🤖 Special Prefixes")
        st.markdown("""
        - `reform` - AI-powered question improvement and suggestions
        - `memory` - Ask about our conversation history and previous topics
        - `base_search` - Standard search without prefix optimizations
        
        **💡 Memory Prefix Examples:**
        - `memory what weapons did we discuss?`
        - `memory summarize our conversation about vehicles`
        - `memory what was the last API reference we looked at?`
        - `memory show me all the code examples from our chat`
        """)
    
    def render_system_info(self):
        st.markdown("## 📊 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 API Status")
            health = self.check_api_health()
            
            if health["status"] == "healthy":
                st.success("✅ API Connected and Healthy")
                if "data" in health:
                    data = health["data"]
                    st.metric("Vector Stores", len(data.get('vector_stores', [])))
            else:
                st.error(f"❌ API Error: {health['message']}")
        
        with col2:
            st.subheader("📊 System Statistics")
            # Auto-load stats
            if st.session_state.system_stats is None:
                st.session_state.system_stats = self.get_system_stats()
            
            if st.session_state.system_stats:
                stats = st.session_state.system_stats
                st.metric("Total Documents", f"{stats.get('total_documents', 0):,}")
                st.metric("Active Sessions", stats.get('active_sessions', 0))
            else:
                st.warning("Unable to load system statistics")
        
        # Performance Information
        st.markdown("## ⚡ Performance Metrics")
        
        st.markdown("""
        ### 📈 System Capabilities
        
        The Arma RAG system processes queries with impressive performance:
        
        - **Database Scale:** 62,517 documents across 3 specialized categories
        - **Retrieval Speed:** 200-300 documents in ~0.14 seconds
        - **Category Distribution:**
          - Source_Code: 23,329 documents
          - Documentation: 7,186 documents
          - API_Reference: 32,002 documents
        """)
    
    def run(self):
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Tutorial & Guide", "📊 System Info"])
        
        with tab1:
            self.render_sidebar()
            self.render_main_chat()
        
        with tab2:
            self.render_tutorial_page()
        
        with tab3:
            self.render_system_info()

if __name__ == "__main__":
    app = ArmaRAGWebUI()
    app.run()
