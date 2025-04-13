import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Union

from app.processors.l0.models import (
    ProcessingStatus,
    FileInfo,
    ChunkInfo,
    DocumentInsights,
    ProcessingResult
)
from app.processors.l0.content_extractor import ContentExtractor
from app.processors.l0.chunker import Chunker
from app.processors.l0.document_analyzer import DocumentAnalyzer
from app.processors.l0.embedding_generator import EmbeddingGenerator
from app.processors.l0.utils import setup_logger, retry, safe_execute

# Set up logger
logger = setup_logger(__name__)

class DocumentProcessor:
    """
    Main orchestrator for the L0 processing pipeline.
    Manages the sequence of operations:
    1. Content extraction
    2. Document analysis
    3. Chunking
    4. Embedding generation
    5. Storage coordination
    """
    
    def __init__(self, 
                 storage_provider: Any,  # Should be a storage interface
                 vector_db_provider: Any,  # Should be a vector DB interface
                 openai_api_key: Optional[str] = None,
                 chunking_strategy: str = "paragraph",
                 embedding_model: str = "text-embedding-3-small",
                 analysis_model: str = "gpt-3.5-turbo"):
        """
        Initialize the document processor with required components.
        
        Args:
            storage_provider: Provider for storing documents and chunks
            vector_db_provider: Provider for storing embeddings
            openai_api_key: OpenAI API key
            chunking_strategy: Strategy for chunking ("paragraph" or "fixed")
            embedding_model: OpenAI embedding model name
            analysis_model: OpenAI analysis model name
        """
        self.storage_provider = storage_provider
        self.vector_db_provider = vector_db_provider
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize component classes
        self.content_extractor = ContentExtractor()
        self.chunker = Chunker(max_chunk_size=2000, min_chunk_size=100, overlap=50)
        self.document_analyzer = DocumentAnalyzer(api_key=self.openai_api_key, model=analysis_model)
        self.embedding_generator = EmbeddingGenerator(api_key=self.openai_api_key, model=embedding_model)
        
        self.chunking_strategy = chunking_strategy
    
    @retry(max_retries=2)
    def process_document(self, file_info: FileInfo) -> ProcessingResult:
        """
        Process a document through the entire pipeline.
        
        Args:
            file_info: Document metadata and content
            
        Returns:
            ProcessingResult object with status and insights
        """
        document_id = file_info.document_id
        logger.info(f"Starting processing for document {document_id}: {file_info.filename}")
        
        # Create in-progress result
        result = ProcessingResult.in_progress(document_id)
        
        try:
            # Step 1: Upload original document to storage
            logger.info(f"Uploading original document to storage: {file_info.filename}")
            self._store_original_document(file_info)
            
            # Step 2: Extract content
            logger.info(f"Extracting content from document: {file_info.filename}")
            extracted_content = self.content_extractor.extract(file_info)
            
            # Step 3: Analyze document for insights
            logger.info(f"Analyzing document content for insights: {file_info.filename}")
            insights = self.document_analyzer.analyze(extracted_content, file_info.filename)
            
            # Step 4: Chunk the content
            logger.info(f"Chunking document content: {file_info.filename}")
            chunks = self._chunk_content(extracted_content, document_id)
            
            # Step 5: Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Step 6: Store chunks and their embeddings
            logger.info(f"Storing {len(chunks_with_embeddings)} chunks and embeddings")
            stored_chunks = self._store_chunks_and_embeddings(chunks_with_embeddings, file_info)
            
            # Create success result
            result = ProcessingResult.success(
                document_id=document_id,
                chunk_count=len(stored_chunks),
                insights=insights
            )
            
            logger.info(f"Successfully processed document {document_id}: {file_info.filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return ProcessingResult.failure(document_id, str(e))
    
    def _chunk_content(self, content: str, document_id: str) -> List[ChunkInfo]:
        """
        Chunk the document content using the appropriate strategy.
        
        Args:
            content: Extracted document content
            document_id: Document ID
            
        Returns:
            List of ChunkInfo objects
        """
        metadata = {
            "document_id": document_id
        }
        
        if self.chunking_strategy == "paragraph":
            return self.chunker.chunk_by_paragraph(content, metadata)
        else:
            return self.chunker.chunk_by_fixed_size(content, metadata)
    
    def _store_original_document(self, file_info: FileInfo) -> str:
        """
        Store the original document in the storage provider.
        
        Args:
            file_info: Document metadata and content
            
        Returns:
            S3 path where the document was stored
        """
        # This should call the storage provider to store the document
        # Implement based on your storage provider interface
        return self.storage_provider.store_document(file_info)
    
    def _store_chunks_and_embeddings(self, chunks: List[ChunkInfo], file_info: FileInfo) -> List[ChunkInfo]:
        """
        Store chunks in storage provider and embeddings in vector DB.
        
        Args:
            chunks: List of chunks with embeddings
            file_info: Original document metadata
            
        Returns:
            List of chunks after storage
        """
        # Store each chunk as a separate file
        for i, chunk in enumerate(chunks):
            # Generate a unique ID for the chunk if not present
            if not hasattr(chunk, 'chunk_id') or not chunk.chunk_id:
                chunk.chunk_id = f"{file_info.document_id}_chunk_{i}"
            
            # Generate S3 path for the chunk
            chunk.s3_path = f"chunks/{file_info.document_id}/chunk_{i}.txt"
            
            # Set the document_id and chunk_index
            chunk.document_id = file_info.document_id
            chunk.chunk_index = i
            
            # Add metadata if not present
            if not chunk.metadata:
                chunk.metadata = {}
            
            # Add additional metadata
            chunk.metadata.update({
                "filename": file_info.filename,
                "content_type": file_info.content_type,
                "chunk_index": i
            })
            
            # Store the chunk content in storage provider
            self.storage_provider.store_chunk(chunk)
            
        # Store all embeddings in vector DB
        self.vector_db_provider.store_embeddings(chunks)
        
        return chunks
