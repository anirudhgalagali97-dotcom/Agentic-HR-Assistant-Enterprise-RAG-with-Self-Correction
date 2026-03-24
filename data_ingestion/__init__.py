"""Data Ingestion package for Agentic RAG System."""
from .ingest import DocumentIngester, create_ingester, run_ingestion

__all__ = ["DocumentIngester", "create_ingester", "run_ingestion"]
