import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union

from app.processors.l0.models import (
    ProcessingStatus,
    FileInfo,
    ChunkInfo,
    DocumentInsight,
    DocumentSummary,
    ProcessingResult
)
from app.processors.l0.content_extractor import ContentExtractor
from app.processors.l0.chunker import Chunker
from app.processors.l0.document_analyzer import DocumentAnalyzer
from app.processors.l0.document_insight_processor import DocumentInsightProcessor
from app.processors.l0.embedding_generator import EmbeddingGenerator
from app.processors.l0.utils import setup_logger, retry, safe_execute

# Set up logger
logger = setup_logger(__name__)

class DocumentProcessor:
    """
    Main orchestrator for the L0 processing pipeline.
    Manages the sequence of operations:
    1. Content extraction
    2. Document analysis (two-stage: insight and summary)
    3. Chunking
    4. Embedding generation
    5. Storage coordination
    """
    
    # Default tenant ID for MVP
    DEFAULT_TENANT_ID = "1"
    
    def __init__(self, 
                 storage_provider: Any,  # BlobStore
                 vector_db_provider: Any,  # VectorDB
                 rel_db_provider: Any,  # RelationalDB
                 openai_api_key: Optional[str] = None,
                 chunking_strategy: str = "paragraph",
                 embedding_model: str = "text-embedding-3-small",
                 insight_model: str = "gpt-4o-mini",
                 summary_model: str = "gpt-3.5-turbo",
                 user_id: str = "1"):  # Default user_id for MVP
        """
        Initialize the document processor with required components.
        
        Args:
            storage_provider: Provider for storing documents and chunks (BlobStore)
            vector_db_provider: Provider for storing embeddings (VectorDB)
            rel_db_provider: Provider for storing structured data (RelationalDB)
            openai_api_key: OpenAI API key
            chunking_strategy: Strategy for chunking ("paragraph" or "fixed")
            embedding_model: OpenAI embedding model name
            insight_model: OpenAI model for generating deep insights
            summary_model: OpenAI model for generating summaries
            user_id: User ID (defaults to "1" for MVP)
        """
        self.storage_provider = storage_provider
        self.vector_db_provider = vector_db_provider
        self.rel_db_provider = rel_db_provider
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.user_id = user_id
        
        # For MVP, warn if user_id is not the default since we'll use DEFAULT_TENANT_ID for vector DB
        if self.user_id != self.DEFAULT_TENANT_ID:
            logger.warning(f"Using non-default user_id '{user_id}' for storage paths, but vector DB operations will use tenant_id='{self.DEFAULT_TENANT_ID}' for MVP")
        
        # Ensure default tenant exists in vector database
        try:
            self._ensure_tenant_exists(self.DEFAULT_TENANT_ID)
        except Exception as e:
            logger.warning(f"Could not verify tenant '{self.DEFAULT_TENANT_ID}' during initialization. Will try again when processing: {e}")
        
        # Initialize component classes
        self.content_extractor = ContentExtractor()
        self.chunker = Chunker(max_chunk_size=1000, min_chunk_size=100, overlap=50)
        
        # Two-stage document analysis processor
        self.document_insight_processor = DocumentInsightProcessor(
            api_key=self.openai_api_key,
            insight_model=insight_model,
            summary_model=summary_model
        )
        
        self.embedding_generator = EmbeddingGenerator(api_key=self.openai_api_key, model=embedding_model)
        
        self.chunking_strategy = chunking_strategy
    
    def _ensure_tenant_exists(self, tenant_id: str) -> None:
        """
        Ensure that the specified tenant exists in the vector database.
        Creates the tenant if it doesn't exist.
        
        Args:
            tenant_id: Tenant ID to check/create
        """
        try:
            # Check if the tenant already exists
            tenants = self.vector_db_provider.list_tenants()
            
            if tenant_id not in tenants:
                logger.info(f"Creating tenant {tenant_id} in vector database")
                success = self.vector_db_provider.create_tenant(tenant_id)
                if success:
                    logger.info(f"Successfully created tenant {tenant_id}")
                else:
                    logger.warning(f"Failed to create tenant {tenant_id}")
        except Exception as e:
            logger.error(f"Error ensuring tenant {tenant_id} exists: {e}")
            raise
    
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
        
        # Get DB session
        db_session = self.rel_db_provider.get_db_session()
        
        try:
            # Update document status to "processing" in the database
            self._update_document_status(db_session, document_id, ProcessingStatus.PROCESSING)
            
            # Step 1: Upload original document to storage (Wasabi)
            s3_path = self._store_original_document(file_info)
            logger.info(f"Uploading original document to storage: {file_info.filename} to {s3_path}")
            
            # Step 2: Extract content
            logger.info(f"Extracting content from document: {file_info.filename}")
            extracted_content = self.content_extractor.extract(file_info)
            
            # Step 3: Analyze document using two-stage process
            logger.info(f"Using two-stage analysis for document: {file_info.filename}")
            analysis_results = self.document_insight_processor.process_document(
                content=extracted_content,
                filename=file_info.filename,
                document_id=document_id
            )
            insight = analysis_results["insight"]
            summary = analysis_results["summary"]
            
            # Store both insight and summary
            logger.info(f"Storing document insight and summary")
            insight_path = self._store_insight(document_id, insight)
            summary_path = self._store_summary(document_id, summary)
            
            # Step 4: Chunk the content
            logger.info(f"Chunking document content: {file_info.filename}")
            chunks = self._chunk_content(extracted_content, document_id)
            
            # Step 5: Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Step 6: Store chunks and their embeddings
            logger.info(f"Storing {len(chunks_with_embeddings)} chunks and embeddings")
            stored_chunks = self._store_chunks_and_embeddings(chunks_with_embeddings, file_info)
            
            # Step 7: Update document status to "completed" in the database
            self._update_document_status(
                db_session, 
                document_id, 
                ProcessingStatus.COMPLETED, 
                len(stored_chunks)
            )
            
            # Create success result
            result = ProcessingResult.success(
                document_id=document_id,
                chunk_count=len(stored_chunks),
                insight=insight,
                summary=summary
            )
            
            logger.info(f"Successfully processed document {document_id}: {file_info.filename}")
            db_session.commit()
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            db_session.rollback()
            
            # Update document status to "failed" in the database
            try:
                self._update_document_status(
                    db_session, 
                    document_id, 
                    ProcessingStatus.FAILED,
                    error_message=str(e)
                )
                db_session.commit()
            except:
                logger.error("Failed to update document status to failed")
                
            return ProcessingResult.failure(document_id, str(e))
        finally:
            # Close the database session
            self.rel_db_provider.close_db_session(db_session)
    
    def _update_document_status(self, 
                               db_session, 
                               document_id: str, 
                               status: ProcessingStatus,
                               chunk_count: int = 0,
                               error_message: Optional[str] = None) -> None:
        """
        Update the document processing status in the database.
        
        Args:
            db_session: Database session
            document_id: Document ID
            status: New processing status
            chunk_count: Number of chunks (for completed documents)
            error_message: Error message (for failed documents)
        """
        # Get status as string
        status_str = status.value
        
        if status == ProcessingStatus.COMPLETED:
            # Update document as processed with chunk count
            self.rel_db_provider.update_document_processed(
                session=db_session,
                document_id=document_id,
                processed=True,
                chunk_count=chunk_count
            )
        elif status == ProcessingStatus.FAILED:
            # If we have a training job table, we could log errors there
            # For now, just mark as not processed
            self.rel_db_provider.update_document_processed(
                session=db_session,
                document_id=document_id,
                processed=False,
                chunk_count=0
            )
    
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
        
        # Get chunks as dictionaries from the chunker
        if self.chunking_strategy == "paragraph":
            chunk_dicts = self.chunker.chunk_by_paragraph(content, document_id, metadata=metadata)
        else:
            chunk_dicts = self.chunker.chunk_by_fixed_size(content, document_id, metadata=metadata)
        
        # Convert dictionaries to ChunkInfo objects
        chunks = []
        for chunk_dict in chunk_dicts:
            chunk = ChunkInfo(
                chunk_id=chunk_dict.get("chunk_id", ""),
                document_id=document_id,
                chunk_index=chunk_dict.get("chunk_index", 0),
                content=chunk_dict.get("content", ""),
                s3_path="",  # Will be set later
                metadata=chunk_dict.get("metadata", {})
            )
            chunks.append(chunk)
        
        return chunks
    
    def _store_original_document(self, file_info: FileInfo) -> str:
        """
        Store the original document in the storage provider.
        
        Args:
            file_info: Document metadata and content
            
        Returns:
            S3 path where the document was stored
        """
        # Create S3 key using tenant ID (user_id) and filename
        s3_key = f"tenant/{self.user_id}/raw/{file_info.document_id}_{file_info.filename}"
        
        # Create metadata for the file
        metadata = {
            "document_id": str(file_info.document_id),
            "content_type": file_info.content_type,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store the document using BlobStore
        if file_info.content:
            # If content is provided as bytes
            s3_uri = self.storage_provider.put_object(s3_key, file_info.content, metadata)
        else:
            # If no content is provided, we assume it's a reference to a local file
            s3_uri = self.storage_provider.put_file(s3_key, file_info.s3_path, metadata)
        
        # Return the S3 path
        return s3_uri
    
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
            try:
                # Add debugging information
                logger.info(f"Processing chunk {i} with content type: {type(chunk.content)}")
                
                # Generate a unique ID for the chunk if not present
                if not hasattr(chunk, 'chunk_id') or not chunk.chunk_id:
                    chunk.chunk_id = f"{file_info.document_id}_chunk_{i}"
                
                # Generate S3 path for the chunk
                s3_key = f"tenant/{self.user_id}/chunks/{file_info.document_id}/chunk_{i}.txt"
                chunk.s3_path = s3_key
                
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
                    "chunk_index": i,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Ensure content is a string before encoding
                if chunk.content is None:
                    logger.warning(f"Chunk {i} has None content")
                    content_str = ""
                elif isinstance(chunk.content, (int, float, bool)):
                    logger.warning(f"Chunk {i} has non-string content: {type(chunk.content)}")
                    content_str = str(chunk.content)
                elif isinstance(chunk.content, str):
                    content_str = chunk.content
                else:
                    logger.warning(f"Chunk {i} has unexpected content type: {type(chunk.content)}")
                    content_str = str(chunk.content)
                
                # Store the chunk content in storage provider
                logger.info(f"Storing chunk {i} content to S3 (length: {len(content_str)})")
                
                # Convert metadata values to strings to ensure compatibility with the storage provider
                string_metadata = {}
                if chunk.metadata:
                    string_metadata = {k: str(v) for k, v in chunk.metadata.items()}
                
                self.storage_provider.put_object(
                    s3_key,
                    content_str.encode('utf-8'), # store the chunk content
                    string_metadata
                )
                
                # Add the chunk to vector DB
                if chunk.embedding:
                    logger.info(f"Adding chunk {i} to vector DB (embedding dim: {len(chunk.embedding)})")
                    
                    # Make sure tenant exists before adding chunk
                    tenant_id = self.DEFAULT_TENANT_ID  # Fixed tenant ID for MVP
                    self._ensure_tenant_exists(tenant_id)
                    
                    # Add the chunk to vector DB
                    self.vector_db_provider.add_chunk(
                        tenant_id=tenant_id,  # Fixed tenant ID for MVP
                        document_id=chunk.document_id,
                        s3_path=chunk.s3_path,
                        chunk_index=chunk.chunk_index,
                        embedding=chunk.embedding,
                        metadata=chunk.metadata
                    )
                else:
                    logger.warning(f"Chunk {i} for document {file_info.document_id} has no embedding")
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                # logger.error(f"Chunk content type: {type(chunk.content)}")
                # logger.error(f"Chunk content: {chunk.content}")
                raise
        
        return chunks
    
    def _store_insight(self, document_id: str, insight: DocumentInsight) -> str:
        """
        Store document insight in the storage provider.
        
        Args:
            document_id: Document ID
            insight: Document insight
            
        Returns:
            S3 path where insight was stored
        """
        import json
        
        # Create insight object
        insight_dict = {
            "title": insight.title,
            "insight": insight.insight,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Convert to JSON
        insight_json = json.dumps(insight_dict, ensure_ascii=False, indent=2)
        
        # Store in S3
        s3_key = f"tenant/{self.user_id}/metadata/{document_id}/insight.json"
        s3_uri = self.storage_provider.put_object(
            s3_key,
            insight_json.encode('utf-8'),
            {"document_id": str(document_id)}
        )
        
        return s3_uri
    
    def _store_summary(self, document_id: str, summary: DocumentSummary) -> str:
        """
        Store document summary in the storage provider.
        
        Args:
            document_id: Document ID
            summary: Document summary
            
        Returns:
            S3 path where summary was stored
        """
        import json
        
        # Create summary object
        summary_dict = {
            "title": summary.title,
            "summary": summary.summary,
            "keywords": summary.keywords,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Convert to JSON
        summary_json = json.dumps(summary_dict, ensure_ascii=False, indent=2)
        
        # Store in S3
        s3_key = f"tenant/{self.user_id}/metadata/{document_id}/summary.json"
        s3_uri = self.storage_provider.put_object(
            s3_key,
            summary_json.encode('utf-8'),
            {"document_id": str(document_id)}
        )
        
        return s3_uri
