"""
Self-Querying Retriever for Agentic RAG System
Uses LLM to extract metadata filters and query from natural language
"""
from typing import List, Optional, Dict, Any, Callable
import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.structured_query import (
    StructuredQuery,
    Operator,
    Comparator,
    visit_structured_query
)
from langchain_core.structured_query import (
    parse_structured_query
)
from langchain.retrievers.self_query.base import SelfQueryRetriever as LCSelfQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo

from config.settings import settings


logger = logging.getLogger(__name__)


# Metadata field descriptions for document schema
DOCUMENT_SCHEMA = [
    AttributeInfo(
        name="source",
        description="The file path or URL where the document came from",
        type="string"
    ),
    AttributeInfo(
        name="file_name",
        description="The name of the PDF file",
        type="string"
    ),
    AttributeInfo(
        name="file_hash",
        description="SHA256 hash of the source file for deduplication",
        type="string"
    ),
    AttributeInfo(
        name="loaded_at",
        description="Timestamp when the document was loaded",
        type="string"
    ),
    AttributeInfo(
        name="chunk_id",
        description="Index of the chunk within the original document",
        type="integer"
    ),
    AttributeInfo(
        name="total_chunks",
        description="Total number of chunks in the document",
        type="integer"
    ),
    AttributeInfo(
        name="page",
        description="Page number in the original PDF (if available)",
        type="integer"
    ),
]


class SelfQueryRetrieverWrapper:
    """Wrapper around LangChain's SelfQueryRetriever with custom enhancements."""
    
    def __init__(
        self,
        vectorstore: Any,
        llm: Any,
        document_content_description: str = "Scientific or technical documents",
        metadata_field_info: List[AttributeInfo] = None,
        enable_limit: bool = True,
        search_kwargs: dict = None
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.document_content_description = document_content_description
        self.metadata_field_info = metadata_field_info or DOCUMENT_SCHEMA
        self.search_kwargs = search_kwargs or {"k": settings.vector_search_k}
        
        self.retriever = LCSelfQueryRetriever(
            vectorstore=vectorstore,
            llm=llm,
            document_content_description=document_content_description,
            metadata_field_info=self.metadata_field_info,
            enable_limit=enable_limit,
            search_kwargs=self.search_kwargs,
            verbose=True
        )
    
    def invoke(self, query: str) -> List[Document]:
        """Invoke the retriever with a query."""
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error in self-query retrieval: {str(e)}")
            # Fallback to simple similarity search
            return self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 5))
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async invoke the retriever."""
        try:
            return await self.retriever.ainvoke(query)
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
