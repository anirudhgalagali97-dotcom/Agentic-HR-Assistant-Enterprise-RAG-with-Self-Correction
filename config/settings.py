"""
Configuration Management for Agentic RAG System
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # In Docker, WORKDIR is /app
    # Locally, use the directory containing this file
    return Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API Key for web search")
    
    # Model Configuration
    llm_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    
    # Vector Store Configuration (relative to project root)
    @property
    def vector_store_path(self) -> Path:
        return get_project_root() / "data" / "chroma_db"
    
    collection_name: str = "agentic_rag_docs"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_file_types: list = [".pdf"]
    
    # Retrieval Configuration
    vector_search_k: int = 5
    bm25_k: int = 5
    ensemble_weights: list = [0.5, 0.5]  # [bm25, vector]
    min_relevance_score: float = 0.5
    
    # Agent Configuration
    max_web_search_results: int = 3
    max_retrieval_iterations: int = 3
    hallucination_threshold: float = 0.7
    
    # Observability
    log_level: str = "INFO"
    enable_tracing: bool = True
    
    @property
    def log_file_path(self) -> Path:
        return get_project_root() / "logs" / "rag_observability.log"
    
    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Gradio Configuration
    gradio_share: bool = False
    gradio_port: int = 7860
    
    # Data Paths (relative to project root)
    @property
    def data_dir(self) -> Path:
        return get_project_root() / "data" / "documents"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Ensure directories exist
def init_directories():
    """Initialize required directories."""
    # Get the actual Path objects by accessing the properties
    vector_path = Path(settings.vector_store_path)
    data_path = Path(settings.data_dir)
    log_path = Path(settings.log_file_path)
    
    vector_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)


# Environment variable helpers
def get_openai_api_key() -> str:
    """Get OpenAI API key from environment or settings."""
    api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Please set it in .env file or environment variable.\n"
            "Example: echo 'OPENAI_API_KEY=sk-...' > .env"
        )
    return api_key


def get_tavily_api_key() -> Optional[str]:
    """Get Tavily API key from environment or settings."""
    return os.getenv("TAVILY_API_KEY", settings.tavily_api_key)


# For debugging - print paths
def print_paths():
    """Print all configured paths for debugging."""
    print(f"Project root: {get_project_root()}")
    print(f"Vector store path: {settings.vector_store_path}")
    print(f"Data directory: {settings.data_dir}")
    print(f"Log file path: {settings.log_file_path}")
