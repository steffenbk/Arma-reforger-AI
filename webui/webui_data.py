"""
Data and Configuration for Arma RAG WebUI
Contains prefix information, API endpoints, and static data
"""

def get_prefix_info():
    """Returns detailed information about all available prefixes"""
    return {
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
        "base_search": {
            "name": "Base Search",
            "docs": "Default system behavior",
            "time": "Variable",
            "description": "Standard search without prefix optimizations",
            "color": "secondary"
        }
    }

def get_prefix_categories():
    """Returns the categorized prefix structure for the sidebar"""
    return {
        "⚡ Quick": ["quick_doc", "quick_code", "quick_api", "quick_all"],
        "🎯 Standard": ["standard_doc", "standard_code", "standard_api", "standard_all"],
        "🚀 Force": ["force_doc", "force_code", "force_api", "force_all"],
        "🧠 Dynamic": ["dynamic_doc", "dynamic_code", "dynamic_api", "dynamic_all"],
        "🤖 Special": ["reform", "base_search"]
    }

class APIConfig:
    """Configuration for API endpoints and settings"""
    BASE_URL = "http://localhost:8000"
    
    ENDPOINTS = {
        "health": "/health",
        "stats": "/stats", 
        "chat": "/chat",
        "conversation_clear": "/conversation/{session_id}/clear"
    }
    
    TIMEOUTS = {
        "health": 5,
        "stats": 10,
        "chat": 300,
        "conversation_clear": 10
    }

def get_welcome_content():
    """Returns the welcome message content"""
    return {
        "title": "👋 Welcome to Arma Reforger RAG Assistant!",
        "description": """
        Ask me anything about Arma Reforger modding! I have access to comprehensive documentation covering:
        
        - **🔧 Modding Tutorials** - Step-by-step guides and workflows
        - **💻 Source Code** - Real examples and implementations from the game
        - **📖 API Reference** - Complete class documentation and method details
        """,
        "examples": [
            {
                "code": "quick_doc what is Enfusion Script?",
                "description": "Fast introduction to scripting basics"
            },
            {
                "code": "standard_code weapon creation example", 
                "description": "Balanced code examples for weapons"
            },
            {
                "code": "force_api SCR_WeaponComponent documentation",
                "description": "Complete API reference"
            }
        ]
    }