"""
Self-Querying Retriever for Agentic RAG System
Uses LLM to extract metadata filters and query from natural language
"""
from typing import List, Optional, Dict, Any, Callable
import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
# Note: SelfQueryRetriever has been removed in newer LangChain versions
# Using a simplified implementation instead
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


logger = logging.getLogger(__name__)


# Metadata field descriptions for document schema (commented out - not used in simplified version)
# DOCUMENT_SCHEMA = [
#     AttributeInfo(
#         name="source",
#         description="The file path or URL where the document came from",
#         type="string"
#     ),
#     AttributeInfo(
#         name="file_name",
#         description="The name of the PDF file",
#         type="string"
#     ),
#     AttributeInfo(
#         name="file_hash",
#         description="SHA256 hash of the source file for deduplication",
#         type="string"
#     ),
#     AttributeInfo(
#         name="loaded_at",
#         description="Timestamp when the document was loaded",
#         type="string"
#     ),
#     AttributeInfo(
#         name="chunk_id",
#         description="Index of the chunk within the original document",
#         type="integer"
#     ),
#     AttributeInfo(
#         name="total_chunks",
#         description="Total number of chunks in the document",
#         type="integer"
#     ),
#     AttributeInfo(
#         name="page",
#         description="Page number in the original PDF (if available)",
#         type="integer"
#     ),
# ]


class SelfQueryRetrieverWrapper:
    """Wrapper around vectorstore similarity search (replaces SelfQueryRetriever)."""

    def __init__(
        self,
        vectorstore: Any,
        llm: Any = None,  # Not used in simplified version
        document_content_description: str = "Scientific or technical documents",
        metadata_field_info: List = None,  # Not used in simplified version
        enable_limit: bool = True,
        search_kwargs: dict = None
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.document_content_description = document_content_description
        self.metadata_field_info = metadata_field_info
        self.search_kwargs = search_kwargs or {"k": settings.vector_search_k}

        # Use simple similarity search instead of SelfQueryRetriever
        logger.info("Using simplified retriever (SelfQueryRetriever not available in this LangChain version)")

    def invoke(self, query: str) -> List[Document]:
        """Invoke the retriever with a query using similarity search."""
        try:
            return self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 5))
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []

    async def ainvoke(self, query: str) -> List[Document]:
        """Async invoke the retriever using similarity search."""
        try:
            # For async, we'll use the sync method since Chroma doesn't have async similarity_search
            return self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 5))
        except Exception as e:
            logger.error(f"Error in async self-query retrieval: {str(e)}")
            return self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 5))
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous method for backwards compatibility."""
        return self.invoke(query)


class QueryParser:
    """Parse and decompose complex queries."""
    
    def __init__(self, llm: Any):
        self.llm = llm
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query into components."""
        prompt = f"""Analyze the following query and extract:
        1. The main search intent
        2. Any metadata filters (file names, dates, page numbers)
        3. Whether a web search might be needed
        
        Query: {query}
        
        Return a JSON object with keys: intent, filters (dict), needs_web_search (bool)"""
        
        try:
            response = self.llm.invoke(prompt)
            import json
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"Query parsing failed: {e}")
            return {
                "intent": query,
                "filters": {},
                "needs_web_search": False
            }
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries."""
        prompt = f"""Decompose this complex query into simpler sub-queries that can be answered independently.
        
        Query: {query}
        
        Return a JSON array of sub-queries."""
        
        try:
            response = self.llm.invoke(prompt)
            import json
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]


def create_self_query_retriever(
    vectorstore: Any,
    llm: Any,
    **kwargs
) -> SelfQueryRetrieverWrapper:
    """Factory function to create a self-querying retriever."""
    return SelfQueryRetrieverWrapper(
        vectorstore=vectorstore,
        llm=llm,
        **kwargs
    )
