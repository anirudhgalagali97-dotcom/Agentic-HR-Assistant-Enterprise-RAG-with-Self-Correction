"""
Professional-Grade Data Ingestion Pipeline for Agentic RAG System
================================================================

This pipeline implements:
1. Metadata Enrichment - Adds filename, page number, summary tags to each chunk
2. Persistent Storage - Prevents re-embedding costs on every run
3. Duplicate Detection - Hash-based deduplication to avoid cluttering vector space

Key Features:
- Recursive Character Splitting that maintains paragraph/sentence integrity
- Automatic metadata extraction from PDF documents
- File hash-based deduplication
- Batch processing for large document sets
- Progress tracking and comprehensive logging
"""
import os
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Enhanced metadata schema for document chunks."""
    source: str
    file_name: str
    file_hash: str
    page_number: int
    total_pages: int
    chunk_id: int
    total_chunks: int
    loaded_at: str
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    summary_tag: Optional[str] = None
    semantic_category: Optional[str] = None


class ProfessionalDocumentIngester:
    """
    Professional-grade document ingestion pipeline.
    
    Features:
    - Metadata enrichment with file info, page numbers, and tags
    - Duplicate detection via file hashing
    - Persistent vector storage
    - Batch processing for scalability
    """
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_openai_embeddings: bool = False
    ):
        """
        Initialize the professional document ingester.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Embedding model to use
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            use_openai_embeddings: Use OpenAI embeddings if True, otherwise HuggingFace
        """
        self.collection_name = collection_name or settings.collection_name
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.use_openai_embeddings = use_openai_embeddings
        
        # Initialize embeddings
        if use_openai_embeddings:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.openai_api_key
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        # Initialize text splitter with semantic boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Major section breaks
                "\n\n",    # Paragraph breaks (preferred)
                "\n",      # Line breaks
                ". ",      # Sentence boundaries
                ", ",      # Clause boundaries
                " ",       # Word boundaries
                ""         # Character boundaries (fallback)
            ],
            add_start_index=True,
            strip_whitespace=True
        )
        
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()
        
        # Track processed files for deduplication
        self._processed_hashes: set = set()
        self._load_processed_hashes()
    
    def _initialize_vector_store(self):
        """Initialize or load existing ChromaDB vector store."""
        persist_dir = str(settings.vector_store_path)
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            logger.info(f"Loading existing vector store from {persist_dir}")
            self.vector_store = Chroma(
                client=None,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
        else:
            logger.info(f"Creating new vector store at {persist_dir}")
            os.makedirs(persist_dir, exist_ok=True)
            self.vector_store = Chroma(
                client=None,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
    
    def _load_processed_hashes(self):
        """Load hashes of already processed files."""
        hash_file = settings.vector_store_path / ".processed_hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    self._processed_hashes = set(json.load(f))
                logger.info(f"Loaded {len(self._processed_hashes)} processed file hashes")
            except Exception as e:
                logger.warning(f"Could not load hash file: {e}")
                self._processed_hashes = set()
    
    def _save_processed_hashes(self):
        """Save hashes of processed files."""
        hash_file = settings.vector_store_path / ".processed_hashes.json"
        try:
            with open(hash_file, 'w') as f:
                json.dump(list(self._processed_hashes), f)
        except Exception as e:
            logger.warning(f"Could not save hash file: {e}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_document_type(self, file_name: str) -> str:
        """Extract document type from filename."""
        file_lower = file_name.lower()
        if any(keyword in file_lower for keyword in ['policy', 'regulation', 'compliance']):
            return "policy"
        elif any(keyword in file_lower for keyword in ['report', 'analysis', 'review']):
            return "report"
        elif any(keyword in file_lower for keyword in ['manual', 'guide', 'documentation']):
            return "documentation"
        elif any(keyword in file_lower for keyword in ['contract', 'agreement', 'terms']):
            return "contract"
        elif any(keyword in file_lower for keyword in ['invoice', 'receipt', 'billing']):
            return "financial"
        return "general"
    
    def _extract_document_title(self, file_name: str) -> str:
        """Extract a readable title from filename."""
        # Remove extension and common prefixes
        title = os.path.splitext(file_name)[0]
        # Replace underscores and dashes with spaces
        title = title.replace('_', ' ').replace('-', ' ')
        # Remove common prefixes like dates
        import re
        title = re.sub(r'^\d{4}[-_]\d{2}[-_]\d{2}[-_]', '', title)
        title = re.sub(r'^[a-zA-Z]{3,4}[-_]\d+[-_]', '', title)
        return title.strip()
    
    def is_duplicate(self, file_path: str) -> bool:
        """Check if a file has already been processed."""
        file_hash = self._compute_file_hash(file_path)
        is_dup = file_hash in self._processed_hashes
        if is_dup:
            logger.info(f"Skipping duplicate file: {file_path}")
        return is_dup
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return documents with rich metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with enriched metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check for duplicates
        file_hash = self._compute_file_hash(file_path)
        if file_hash in self._processed_hashes:
            logger.info(f"Skipping duplicate file: {file_path}")
            return []
        
        logger.info(f"Loading PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            file_name = os.path.basename(file_path)
            total_pages = len(documents)
            
            for i, doc in enumerate(documents, 1):
                # Core metadata
                doc.metadata.update({
                    "source": file_path,
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "page_number": i,
                    "total_pages": total_pages,
                    "loaded_at": datetime.now().isoformat(),
                    "document_type": self._extract_document_type(file_name),
                    "document_title": self._extract_document_title(file_name),
                })
            
            logger.info(f"Loaded {len(documents)} pages from {file_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def load_pdfs_from_directory(
        self,
        directory: str,
        recursive: bool = True,
        glob_pattern: str = "**/*.pdf"
    ) -> Tuple[List[Document], Dict[str, str]]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Directory path to search
            recursive: Whether to search subdirectories
            glob_pattern: Pattern to match PDF files
            
        Returns:
            Tuple of (list of documents, dict of file paths to errors)
        """
        all_documents = []
        errors = {}
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return all_documents, errors
        
        # Find all PDF files
        if recursive:
            pdf_files = list(directory_path.glob(glob_pattern))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_file in pdf_files:
            try:
                documents = self.load_pdf(str(pdf_file))
                if documents:  # Only add if not a duplicate
                    all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                errors[str(pdf_file)] = str(e)
        
        return all_documents, errors
    
    def split_documents(
        self,
        documents: List[Document],
        preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects
            preserve_metadata: Whether to copy metadata to chunks
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        # Perform semantic splitting
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
            # Ensure key metadata is present
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = "unknown"
            if "file_name" not in chunk.metadata:
                chunk.metadata["file_name"] = "unknown"
            if "file_hash" not in chunk.metadata:
                chunk.metadata["file_hash"] = "unknown"
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def add_documents_to_vector_store(
        self,
        documents: List[Document],
        batch_size: int = 100,
        regenerate_embeddings: bool = False
    ) -> Tuple[int, List[str], List[str]]:
        """
        Add documents to the vector store in batches.
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents per batch
            regenerate_embeddings: Whether to regenerate embeddings
            
        Returns:
            Tuple of (number added, list of IDs, list of errors)
        """
        if not documents:
            logger.warning("No documents to add")
            return 0, [], []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        doc_ids = []
        errors = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            try:
                if regenerate_embeddings:
                    # Get texts and metadata separately
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    ids = self.vector_store.add_texts(
                        texts=texts,
                        metadatas=metadatas,
                        embedding=self.embeddings
                    )
                else:
                    ids = self.vector_store.add_documents(documents=batch)
                
                doc_ids.extend(ids)
                logger.info(f"Added batch {batch_num}/{total_batches}")
                
            except Exception as e:
                error_msg = f"Error in batch {batch_num}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Persist to disk
        self.vector_store.persist()
        logger.info(f"Successfully added {len(doc_ids)} documents to vector store")
        
        return len(doc_ids), doc_ids, errors
    
    def ingest_pdf_file(
        self,
        file_path: str,
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Complete ingestion pipeline for a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            skip_duplicates: Whether to skip duplicate files
            
        Returns:
            Dictionary with ingestion results
        """
        result = {
            "file_path": file_path,
            "success": False,
            "chunks_added": 0,
            "doc_ids": [],
            "errors": []
        }
        
        # Check for duplicates
        file_hash = self._compute_file_hash(file_path)
        if skip_duplicates and file_hash in self._processed_hashes:
            result["message"] = "Skipped - duplicate file"
            return result
        
        try:
            # Load PDF
            documents = self.load_pdf(file_path)
            if not documents:
                result["errors"].append("No documents loaded")
                return result
            
            # Split into chunks
            chunks = self.split_documents(documents)
            
            # Add to vector store
            count, ids, errors = self.add_documents_to_vector_store(chunks)
            
            # Mark as processed
            self._processed_hashes.add(file_hash)
            self._save_processed_hashes()
            
            result.update({
                "success": True,
                "chunks_added": count,
                "doc_ids": ids,
                "errors": errors,
                "file_hash": file_hash
            })
            
            logger.info(f"Ingestion complete for {file_path}: {count} chunks added")
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Ingestion failed for {file_path}: {e}")
        
        return result
    
    def ingest_directory(
        self,
        directory: str,
        skip_duplicates: bool = True,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Complete ingestion pipeline for all PDFs in a directory.
        
        Args:
            directory: Directory path
            skip_duplicates: Whether to skip duplicate files
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with overall ingestion results
        """
        logger.info(f"Starting directory ingestion: {directory}")
        
        documents, errors = self.load_pdfs_from_directory(
            directory,
            recursive=recursive
        )
        
        if not documents:
            return {
                "success": False,
                "message": "No documents loaded",
                "errors": errors
            }
        
        chunks = self.split_documents(documents)
        count, doc_ids, add_errors = self.add_documents_to_vector_store(chunks)
        
        # Save processed hashes
        for doc in documents:
            file_hash = doc.metadata.get("file_hash")
            if file_hash and file_hash not in self._processed_hashes:
                self._processed_hashes.add(file_hash)
        self._save_processed_hashes()
        
        result = {
            "success": True,
            "documents_loaded": len(documents),
            "chunks_added": count,
            "doc_ids": doc_ids,
            "load_errors": errors,
            "add_errors": add_errors
        }
        
        logger.info(f"Directory ingestion complete: {count} chunks from {len(documents)} documents")
        return result
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        if self.vector_store is None:
            return {"status": "error", "message": "Vector store not initialized"}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "status": "ready",
                "document_count": count,
                "collection_name": self.collection_name,
                "embedding_model": settings.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "persist_directory": str(settings.vector_store_path),
                "processed_files": len(self._processed_hashes)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_vector_store(self, clear_hash_cache: bool = True):
        """Clear all documents from the vector store."""
        if self.vector_store:
            self.vector_store.delete_collection()
            self._initialize_vector_store()
            
            if clear_hash_cache:
                self._processed_hashes.clear()
                self._save_processed_hashes()
            
            logger.info("Vector store cleared")
    
    def get_all_documents(self, limit: int = None) -> List[Document]:
        """Retrieve all documents from the vector store."""
        if self.vector_store is None:
            return []
        
        try:
            results = self.vector_store.get()
            documents = []
            
            for i in range(len(results.get("documents", []))):
                doc = Document(
                    page_content=results["documents"][i],
                    metadata=results["metadatas"][i] if "metadatas" in results else {}
                )
                documents.append(doc)
                
                if limit and len(documents) >= limit:
                    break
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete all documents associated with a file hash."""
        if self.vector_store is None:
            return 0
        
        try:
            collection = self.vector_store._collection
            results = collection.get(where={"file_hash": file_hash})
            
            if results and results["ids"]:
                count = len(results["ids"])
                self.vector_store.delete(ids=results["ids"])
                self.vector_store.persist()
                
                # Remove from processed hashes
                if file_hash in self._processed_hashes:
                    self._processed_hashes.remove(file_hash)
                    self._save_processed_hashes()
                
                logger.info(f"Deleted {count} documents with hash {file_hash}")
                return count
            
            return 0
        except Exception as e:
            logger.error(f"Error deleting by hash: {e}")
            return 0


# Alias for backward compatibility
DocumentIngester = ProfessionalDocumentIngester


def create_ingester(**kwargs) -> ProfessionalDocumentIngester:
    """Factory function to create a DocumentIngester."""
    return ProfessionalDocumentIngester(**kwargs)


def run_ingestion(directory: str = None, **kwargs) -> Dict[str, Any]:
    """
    Run the complete ingestion pipeline.
    
    Args:
        directory: Directory containing PDF files (default: settings.data_dir)
        **kwargs: Additional arguments for ProfessionalDocumentIngester
        
    Returns:
        Dictionary with ingestion results
    """
    directory = directory or str(settings.data_dir)
    
    ingester = create_ingester(**kwargs)
    result = ingester.ingest_directory(directory)
    result["stats"] = ingester.get_vector_store_stats()
    
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    result = run_ingestion()
    print(f"\n{'='*50}")
    print("INGESTION COMPLETE")
    print(f"{'='*50}")
    print(f"Documents loaded: {result.get('documents_loaded', 0)}")
    print(f"Chunks added: {result.get('chunks_added', 0)}")
    print(f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}")
    
    if result.get('load_errors'):
        print(f"\nLoad Errors: {len(result['load_errors'])}")
    if result.get('add_errors'):
        print(f"Add Errors: {len(result['add_errors'])}")
