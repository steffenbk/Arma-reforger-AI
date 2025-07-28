from dataclasses import dataclass

@dataclass
class APIConfig:
    """Configuration for the API"""
    documents_path: str = r"C:\ArmaModdingRAG\Arma_Reforger_RAG_Organized"
    vector_db_path: str = r"C:\ArmaModdingRAG\chroma_db"
    memory_db_path: str = r"C:\ArmaModdingRAG\conversations.db"
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    
    # API settings
    api_host: str = "localhost"
    api_port: int = 8000
    
    # Document retrieval settings - OPTIMIZED FOUR-TIER SYSTEM
    max_docs_per_query: int = 15           # Default for base_search prefix
    
    # Base search prefix - Explicit document search when no tier specified
    base_search_max_docs: int = 15         # Basic document search (balanced)
    
    # Quick prefixes - Ultra-fast, optimized per category type
    quick_doc_max_docs: int = 12           # Documentation: focused, clean answers
    quick_code_max_docs: int = 25          # Code: benefit from pattern examples  
    quick_api_max_docs: int = 20           # API: comprehensive reference coverage
    quick_all_max_docs: int = 18           # Balanced quick search across all
    
    # Standard prefixes - Optimized allocation per category strengths
    standard_doc_max_docs: int = 30        # Documentation sweet spot (was 50)
    standard_code_max_docs: int = 75       # Code patterns without overload (was 500)
    standard_api_max_docs: int = 100       # API comprehensive reference (was 500)
    standard_code_api_max_docs: int = 150  # Split: 75 code + 75 API (was 500)
    standard_all_max_docs: int = 60        # Balanced search (was 50)
    
    # Force prefixes - Maximum power with smart limits
    force_doc_max_docs: int = 50           # Keep current - all Documentation docs
    force_code_max_docs: int = 200         # Reduced from 500 to avoid noise
    force_api_max_docs: int = 300          # API can handle more references
    force_code_api_max_docs: int = 250     # Split: 125 code + 125 API
    force_all_max_docs: int = 100          # Comprehensive search (was 50)
    force_benchmark_max_docs: int = 5      # Keep for testing
    
    # Dynamic prefixes - Adaptive with optimized ceilings
    dynamic_doc_max_docs: int = 30         # Documentation ceiling (was 50)
    dynamic_code_max_docs: int = 100       # Code ceiling (was 500)
    dynamic_api_max_docs: int = 150        # API ceiling (was 500)
    dynamic_code_api_max_docs: int = 125   # Split ceiling (was 500)
    dynamic_all_max_docs: int = 75         # Balanced ceiling (was 50)
    
    # Progressive retrieval settings - Refined for better efficiency
    progressive_initial_batch: int = 8             # Start smaller (was 10)
    progressive_expand_threshold: float = 0.75     # More selective (was 0.7)
    progressive_max_expansions: int = 3            # Keep max rounds
    progressive_expansion_multiplier: float = 1.5  # Gentler growth (was 1.75)
    
    # Memory settings
    max_conversation_length: int = 10
    conversation_summary_threshold: int = 8
    enable_persistent_memory: bool = True