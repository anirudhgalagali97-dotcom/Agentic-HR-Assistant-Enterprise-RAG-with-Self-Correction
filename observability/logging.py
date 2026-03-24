"""
Observability Module for Agentic RAG System
Tracks token usage, latency, context precision, and other metrics
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from functools import wraps
import threading
from collections import defaultdict
import structlog

from config.settings import settings


# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


@dataclass
class TokenUsage:
    """Track token usage for a request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsage":
        return cls(**data)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    question: str
    timestamp: str
    
    # Timing metrics
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    grading_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    
    # Token metrics
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    
    # Quality metrics
    context_precision: float = 0.0
    hallucination_score: float = 0.0
    retrieval_count: int = 0
    relevant_docs_count: int = 0
    
    # Sources
    sources_used: List[str] = field(default_factory=list)
    web_search_used: bool = False
    
    # Status
    status: str = "success"
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "timestamp": self.timestamp,
            "latency": {
                "total_ms": self.total_latency_ms,
                "retrieval_ms": self.retrieval_latency_ms,
                "grading_ms": self.grading_latency_ms,
                "generation_ms": self.generation_latency_ms
            },
            "tokens": self.token_usage.to_dict(),
            "quality": {
                "context_precision": self.context_precision,
                "hallucination_score": self.hallucination_score
            },
            "retrieval": {
                "retrieval_count": self.retrieval_count,
                "relevant_docs_count": self.relevant_docs_count,
                "web_search_used": self.web_search_used
            },
            "sources": self.sources_used,
            "status": self.status,
            "error": self.error
        }


class ObservabilityLogger:
    """Main observability logger for the RAG system."""
    
    def __init__(
        self,
        log_file_path: Optional[Path] = None,
        enable_console_logging: bool = True,
        enable_file_logging: bool = True
    ):
        self.log_file_path = log_file_path or settings.log_file_path
        self.enable_console_logging = enable_console_logging
        self.enable_file_logging = enable_file_logging
        
        self.logger = structlog.get_logger("agentic_rag")
        
        # In-memory metrics store
        self._metrics_store: Dict[str, QueryMetrics] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self._total_queries = 0
        self._total_errors = 0
        self._avg_latency = 0.0
        self._avg_precision = 0.0
        
        # Setup file logging
        if self.enable_file_logging:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file logging."""
        if self.log_file_path:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Also add file handler to Python logging
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
    
    def start_query(self, question: str) -> str:
        """Start tracking a new query."""
        import uuid
        query_id = str(uuid.uuid4())[:8]
        
        metrics = QueryMetrics(
            query_id=query_id,
            question=question[:200],  # Truncate for storage
            timestamp=datetime.now().isoformat()
        )
        
        with self._lock:
            self._metrics_store[query_id] = metrics
        
        self.logger.info(
            "query_started",
            query_id=query_id,
            question_preview=question[:100]
        )
        
        return query_id
    
    def end_query(self, query_id: str, result: Dict[str, Any]):
        """End tracking a query with results."""
        with self._lock:
            if query_id in self._metrics_store:
                metrics = self._metrics_store[query_id]
                
                # Update metrics from result
                metrics.total_latency_ms = result.get("latency", {}).get("total_ms", 0)
                metrics.retrieval_latency_ms = result.get("latency", {}).get("retrieval_ms", 0)
                metrics.grading_latency_ms = result.get("latency", {}).get("grading_ms", 0)
                metrics.generation_latency_ms = result.get("latency", {}).get("generation_ms", 0)
                
                metrics.context_precision = result.get("context_precision", 0)
                metrics.hallucination_score = result.get("hallucination_score", 0)
                metrics.retrieval_count = result.get("iterations", 0)
                metrics.relevant_docs_count = len(result.get("relevant_documents", []))
                metrics.sources_used = result.get("sources", [])
                metrics.web_search_used = len(result.get("web_search_results", [])) > 0
                metrics.status = result.get("status", "unknown")
                metrics.error = result.get("error")
                
                # Update token usage if available
                if "token_usage" in result:
                    metrics.token_usage = TokenUsage(**result["token_usage"])
                
                # Update statistics
                self._total_queries += 1
                if metrics.status == "error":
                    self._total_errors += 1
                
                self._update_averages(metrics)
        
        self.logger.info(
            "query_completed",
            query_id=query_id,
            status=result.get("status"),
            latency_ms=result.get("latency", {}).get("total_ms", 0),
            context_precision=result.get("context_precision", 0)
        )
    
    def _update_averages(self, metrics: QueryMetrics):
        """Update running averages."""
        n = self._total_queries
        self._avg_latency = ((n - 1) * self._avg_latency + metrics.total_latency_ms) / n
        self._avg_precision = ((n - 1) * self._avg_precision + metrics.context_precision) / n
    
    def get_query_metrics(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific query."""
        with self._lock:
            if query_id in self._metrics_store:
                return self._metrics_store[query_id].to_dict()
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        with self._lock:
            return {
                "total_queries": self._total_queries,
                "total_errors": self._total_errors,
                "error_rate": self._total_errors / self._total_queries if self._total_queries > 0 else 0,
                "avg_latency_ms": self._avg_latency,
                "avg_context_precision": self._avg_precision,
                "queries_in_memory": len(self._metrics_store)
            }
    
    def log_retrieval(self, query_id: str, doc_count: int, latency_ms: float):
        """Log retrieval operation."""
        with self._lock:
            if query_id in self._metrics_store:
                self._metrics_store[query_id].retrieval_latency_ms += latency_ms
                self._metrics_store[query_id].retrieval_count += 1
        
        self.logger.info(
            "retrieval_performed",
            query_id=query_id,
            doc_count=doc_count,
            latency_ms=latency_ms
        )
    
    def log_grading(self, query_id: str, relevant_count: int, total_count: int, latency_ms: float):
        """Log grading operation."""
        with self._lock:
            if query_id in self._metrics_store:
                metrics = self._metrics_store[query_id]
                metrics.grading_latency_ms += latency_ms
                metrics.relevant_docs_count = relevant_count
                metrics.context_precision = relevant_count / total_count if total_count > 0 else 0
        
        self.logger.info(
            "grading_performed",
            query_id=query_id,
            relevant_count=relevant_count,
            total_count=total_count,
            precision=metrics.context_precision if query_id in self._metrics_store else 0,
            latency_ms=latency_ms
        )
    
    def log_generation(self, query_id: str, latency_ms: float, token_usage: TokenUsage):
        """Log generation operation."""
        with self._lock:
            if query_id in self._metrics_store:
                self._metrics_store[query_id].generation_latency_ms = latency_ms
                self._metrics_store[query_id].token_usage = token_usage
        
        self.logger.info(
            "generation_performed",
            query_id=query_id,
            latency_ms=latency_ms,
            total_tokens=token_usage.total_tokens
        )
    
    def export_metrics(self, output_path: Path):
        """Export all metrics to a JSON file."""
        with self._lock:
            metrics_data = {
                query_id: metrics.to_dict()
                for query_id, metrics in self._metrics_store.items()
            }
        
        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info("metrics_exported", output_path=str(output_path))


# Global observability instance
_observability_logger: Optional[ObservabilityLogger] = None


def get_observability_logger() -> ObservabilityLogger:
    """Get or create the global observability logger."""
    global _observability_logger
    if _observability_logger is None:
        _observability_logger = ObservabilityLogger()
    return _observability_logger


def track_latency(func):
    """Decorator to track function latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        return result, latency_ms
    return wrapper


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, logger: ObservabilityLogger = None):
        self.name = name
        self.logger = logger or get_observability_logger()
        self.start_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        return False
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0
        return (time.time() - self.start_time) * 1000
