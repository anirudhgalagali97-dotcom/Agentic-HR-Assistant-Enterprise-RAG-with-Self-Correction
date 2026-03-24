"""
FastAPI Backend for Agentic RAG System
Provides REST API endpoints for document management and querying
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import time
import os
from pathlib import Path

from config.settings import settings, init_directories, get_openai_api_key
from data_ingestion.ingest import DocumentIngester, run_ingestion
from agents.graph import AgenticRAGAgent, get_agent, reset_agent
from observability.logging import get_observability_logger, TimingContext

# Initialize directories
init_directories()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="Production-ready Agentic RAG system with LangChain, LangGraph, and OpenAI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize observability
observability = get_observability_logger()

# Global agent instance
agent: Optional[AgenticRAGAgent] = None


# Pydantic models
class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., description="The question to ask")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")
    include_sources: bool = Field(True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    question: str
    sources: List[Dict[str, Any]] = []
    context_precision: float = 0.0
    hallucination_score: float = 0.0
    iterations: int = 0
    latency_ms: float = 0.0
    status: str = "success"
    web_search_used: bool = False


class DocumentStats(BaseModel):
    """Document statistics model."""
    status: str
    document_count: int
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


class SystemStats(BaseModel):
    """System statistics model."""
    total_queries: int
    total_errors: int
    error_rate: float
    avg_latency_ms: float
    avg_context_precision: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    vector_store_ready: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    logger.info("Starting Agentic RAG API...")
    
    try:
        # Verify API key
        api_key = get_openai_api_key()
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Set it in .env file or environment variable.")
        
        # Initialize agent
        agent = get_agent()
        logger.info("Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Agentic RAG API...")
    reset_agent()


# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "Agentic RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        ingester = DocumentIngester()
        stats = ingester.get_vector_store_stats()
        vector_store_ready = stats.get("status") == "ready"
    except Exception:
        vector_store_ready = False
    
    return HealthResponse(
        status="healthy" if vector_store_ready else "degraded",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        vector_store_ready=vector_store_ready
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    This endpoint processes a question through the Agentic RAG pipeline:
    1. Analyzes the query
    2. Retrieves relevant documents
    3. Grades documents for relevance
    4. Optionally performs web search
    5. Generates an answer
    """
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time = time.time()
    query_id = observability.start_query(request.question)
    
    try:
        result = agent.invoke(
            question=request.question,
            thread_id=request.thread_id
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare sources
        sources = []
        if request.include_sources:
            for doc in result.get("relevant_documents", [])[:5]:
                sources.append({
                    "content": doc.page_content[:500],
                    "source": doc.metadata.get("source", "unknown"),
                    "file_name": doc.metadata.get("file_name", "unknown")
                })
            
            for web_result in result.get("web_search_results", [])[:3]:
                sources.append({
                    "title": web_result.get("title", "No title"),
                    "url": web_result.get("url", ""),
                    "snippet": web_result.get("snippet", "")[:500],
                    "source": "web"
                })
        
        # Update observability
        result["latency"] = {
            "total_ms": latency_ms,
            "retrieval_ms": 0,
            "grading_ms": 0,
            "generation_ms": 0
        }
        observability.end_query(query_id, result)
        
        return QueryResponse(
            answer=result.get("answer", ""),
            question=request.question,
            sources=sources,
            context_precision=result.get("context_precision", 0.0),
            hallucination_score=result.get("hallucination_score", 0.0),
            iterations=result.get("iterations", 0),
            latency_ms=latency_ms,
            status=result.get("status", "success"),
            web_search_used=len(result.get("web_search_results", [])) > 0
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        observability.end_query(query_id, {"status": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    """
    Ingest all documents from the data directory.
    
    This runs in the background to avoid blocking the request.
    """
    def run_ingestion_task():
        try:
            result = run_ingestion()
            logger.info(f"Ingestion complete: {result}")
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
    
    background_tasks.add_task(run_ingestion_task)
    
    return {
        "status": "started",
        "message": "Document ingestion started in background"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a single PDF document.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save file temporarily
        file_path = settings.data_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest the document
        ingester = DocumentIngester()
        count, doc_ids = ingester.ingest_pdf_file(str(file_path))
        
        # Clean up
        os.remove(file_path)
        
        return {
            "status": "success",
            "documents_added": count,
            "file_name": file.filename
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/documents", response_model=DocumentStats)
async def get_document_stats():
    """Get document statistics from the vector store."""
    try:
        ingester = DocumentIngester()
        stats = ingester.get_vector_store_stats()
        return DocumentStats(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/system", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics from observability."""
    try:
        stats = observability.get_statistics()
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store."""
    try:
        ingester = DocumentIngester()
        ingester.clear_vector_store()
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_system():
    """Reset the agent and clear cache."""
    try:
        reset_agent()
        global agent
        agent = get_agent()
        return {"status": "success", "message": "System reset complete"}
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
