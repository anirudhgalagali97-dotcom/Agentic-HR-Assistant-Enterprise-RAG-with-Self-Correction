"""
Hybrid Retrieval System combining BM25 and Vector Search
Using EnsembleRetriever for optimal retrieval performance
"""
from typing import List, Optional, Tuple, Any
import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

from config.settings import settings


logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """Custom BM25 Retriever using rank_bm25."""
    
    def __init__(
        self,
        documents: List[Document],
        k: int = 5,
        preprocess_func: callable = None
    ):
        self.documents = documents
        self.k = k
        self.preprocess_func = preprocess_func or self._tokenize
        
        if documents:
            self._initialize_bm25()
        else:
            self.bm25 = None
    
    def _initialize_bm25(self):
        """Initialize BM25 index with tokenized documents."""
        tokenized_docs = [self.preprocess_func(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization function."""
        return text.lower().split()
    
    def _rebuild_index(self):
        """Rebuild BM25 index when documents change."""
        if self.documents:
            self._initialize_bm25()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever and rebuild index."""
        self.documents.extend(documents)
        self._rebuild_index()
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using BM25."""
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index not initialized")
            return []
        
        tokenized_query = self.preprocess_func(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:self.k]
        
        return [self.documents[i] for i in top_indices if scores[i] > 0]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25."""
    
    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        vector_k: int = None,
        bm25_k: int = None
    ):
        self.vectorstore = vectorstore
        self.documents = documents or []
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        vector_k = vector_k or settings.vector_search_k
        bm25_k = bm25_k or settings.bm25_k
        
        self._initialize_retrievers(vector_k, bm25_k)
    
    def _initialize_retrievers(self, vector_k: int, bm25_k: int):
        """Initialize both retrievers."""
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": vector_k}
        )
        
        self.bm25_retriever = BM25Retriever(
            documents=self.documents,
            k=bm25_k
        )
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight]
        )
    
    def invoke(self, query: str) -> List[Document]:
        """Invoke the hybrid retriever."""
        try:
            return self.ensemble_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Ensemble retrieval failed: {str(e)}")
            return self._fallback_retrieval(query)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous retrieval method."""
        return self.invoke(query)
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async retrieval method."""
        return self.invoke(query)
    
    def _fallback_retrieval(self, query: str) -> List[Document]:
        """Fallback to vector search if ensemble fails."""
        logger.warning("Using fallback vector search")
        return self.vectorstore.similarity_search(query, k=settings.vector_search_k)
    
    def update_documents(self, documents: List[Document]):
        """Update documents and rebuild BM25 index."""
        self.documents = documents
        if hasattr(self.bm25_retriever, 'add_documents'):
            self.bm25_retriever.documents = documents
            self.bm25_retriever._rebuild_index()


class VectorStoreManager:
    """Manages vector store and provides retrieval capabilities."""
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.embedding_model = embedding_model or settings.embedding_model
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vectorstore: Optional[Chroma] = None
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load existing vector store or create new one."""
        persist_dir = str(settings.vector_store_path)
        
        if settings.vector_store_path.exists() and list(settings.vector_store_path.iterdir()):
            logger.info(f"Loading vector store from {persist_dir}")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
        else:
            logger.info(f"Creating new vector store at {persist_dir}")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the vector store."""
        if self.vectorstore is None:
            return []
        
        try:
            return self.vectorstore.get().get("documents", [])
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
    
    def create_hybrid_retriever(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> HybridRetriever:
        """Create a hybrid retriever."""
        documents = self.get_all_documents()
        
        return HybridRetriever(
            vectorstore=self.vectorstore,
            documents=documents,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: dict = None
    ) -> List[Document]:
        """Simple similarity search."""
        k = k or settings.vector_search_k
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """Similarity search with scores."""
        k = k or settings.vector_search_k
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "ready",
                "document_count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"status": "error", "error": str(e)}


def create_hybrid_retriever(
    vectorstore: Chroma = None,
    documents: List[Document] = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> HybridRetriever:
    """Factory function to create a hybrid retriever."""
    if vectorstore is None:
        manager = VectorStoreManager()
        vectorstore = manager.vectorstore
        documents = manager.get_all_documents()
    
    return HybridRetriever(
        vectorstore=vectorstore,
        documents=documents,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight
    )
