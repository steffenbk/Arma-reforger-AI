"""
CSS Styles for Arma RAG WebUI
Contains all styling definitions for the Streamlit interface
"""

def get_css_styles():
    """Returns the complete CSS styling for the WebUI"""
    return """
<style>
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    button[kind="header"] {
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
    
    /* Remove any colored left borders from text content */
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
    
    /* Clean, simple styling for all content sections */
    .content-section {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
    }
    
    .content-section h3 {
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
        color: #2c3e50;
    }
    
    .content-section h4 {
        font-size: 1.1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        color: #34495e;
    }
    
    .content-section p, .content-section li {
        font-size: 1rem !important;
        line-height: 1.6 !important;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .content-section ul, .content-section ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .content-section code {
        background-color: rgba(27, 31, 35, 0.1);
        border-radius: 3px;
        font-size: 0.95rem !important;
        padding: 2px 4px;
        color: #d73a49;
        font-weight: 600;
    }
    
    /* Quick Reference styling */
    .quick-reference {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .quick-reference h2 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .quick-reference p {
        color: #6c757d;
        font-style: italic;
        margin-bottom: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""