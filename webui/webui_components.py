"""
UI Components for Arma RAG WebUI
Contains reusable UI components and rendering functions
"""

import streamlit as st
import re
from datetime import datetime
from typing import Dict, List
from webui_data import get_prefix_info, get_prefix_categories, get_welcome_content

class ChatRenderer:
    """Handles chat message rendering and display"""
    
    @staticmethod
    def extract_thinking_content(text: str) -> tuple:
        """Extract thinking content from AI responses"""
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, text, re.DOTALL)
        if matches:
            thinking_content = matches[0].strip()
            clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
            return clean_text, thinking_content
        return text, None
    
    @staticmethod
    def render_code_block(code: str, language: str = ""):
        """Render code blocks with syntax highlighting"""
        st.code(code, language=language)
    
    @staticmethod
    def format_message_content(content: str):
        """Format message content with code blocks"""
        parts = re.split(r'```(\w*)\n(.*?)\n```', content, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if i % 3 == 0:
                if part.strip():
                    st.markdown(part)
            elif i % 3 == 1:
                continue
            else:
                language = parts[i-1] if i > 0 else ""
                ChatRenderer.render_code_block(part, language)
    
    @staticmethod
    def render_chat_message(message: Dict, index: int):
        """Render a single chat message"""
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
                clean_content, thinking_content = ChatRenderer.extract_thinking_content(content)
                ChatRenderer.format_message_content(clean_content)
                
                if thinking_content:
                    thinking_key = f"thinking_{index}"
                    # Initialize the show_reasoning state if not exists
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
                    ProcessingInfoRenderer.render_processing_info(message["processing_info"])
            
            st.markdown('</div>', unsafe_allow_html=True)

class ProcessingInfoRenderer:
    """Handles processing information display"""
    
    @staticmethod
    def render_processing_info(info: Dict):
        """Render processing information metrics"""
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

class PrefixInfoRenderer:
    """Handles prefix information display"""
    
    @staticmethod
    def render_prefix_info(prefix: str):
        """Render detailed information about selected prefix"""
        prefix_info = get_prefix_info()
        
        info = prefix_info.get(prefix, {
            "name": "Unknown Prefix",
            "docs": "Unknown",
            "time": "Unknown", 
            "description": "No information available",
            "color": "secondary"
        })
        
        # Display detailed prefix information
        st.info(f"""
        💡 **Selected: {info['name']}** (`{prefix}`)
        
        📄 **Search Scope:** {info['docs']}  
        ⏱️ **Expected Time:** {info['time']}  
        🎯 **Purpose:** {info['description']}
        
        Type your question and it will be automatically prefixed!
        """)

class SidebarRenderer:
    """Handles sidebar rendering"""
    
    @staticmethod
    def render_system_status():
        """Render system status section"""
        st.subheader("🔧 System Status")
        # This will be implemented by the main class that has API access
        pass
    
    @staticmethod
    def render_prefix_categories():
        """Render prefix category buttons"""
        st.subheader("⚡ Quick Prefixes")
        
        prefix_categories = get_prefix_categories()
        
        for category, prefixes in prefix_categories.items():
            with st.expander(category, expanded=False):
                for prefix in prefixes:
                    if st.button(prefix, key=f"prefix_{prefix}", use_container_width=True):
                        st.session_state.selected_prefix = prefix
    
    @staticmethod
    def render_memory_controls():
        """Render memory control buttons"""
        st.subheader("💾 Memory Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Memory", use_container_width=True):
                return "clear"
        
        with col2:
            if st.button("💾 Save Chat", use_container_width=True):
                return "save"
        
        uploaded_file = st.file_uploader("📤 Import Chat", type=['json'])
        if uploaded_file is not None:
            return uploaded_file
        
        return None

class WelcomeRenderer:
    """Handles welcome screen rendering"""
    
    @staticmethod
    def render_welcome():
        """Render the welcome screen"""
        welcome = get_welcome_content()
        
        st.markdown(f"### {welcome['title']}")
        st.markdown(welcome['description'])
        
        # Quick Start Examples in a styled box
        st.markdown("""
        <div class="tutorial-section">
        <h4>🚀 Quick Start Examples:</h4>
        <ul>
        """, unsafe_allow_html=True)
        
        for example in welcome['examples']:
            st.markdown(f"""
            <li><code>{example['code']}</code> - {example['description']}</li>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        </ul>
        </div>
        """, unsafe_allow_html=True)