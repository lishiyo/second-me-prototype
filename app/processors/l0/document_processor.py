import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import inspect

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
from app.core.config import settings
from app.providers.rel_db import Document

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
    
    def __init__(self, 
                 storage_provider: Any,  # BlobStore
                 vector_db_provider: Any,  # VectorDB
                 rel_db_provider: Any,  # RelationalDB
                 openai_api_key: Optional[str] = None,
                 chunking_strategy: str = "paragraph",
                 embedding_model: str = "text-embedding-3-small",
                 insight_model: str = "gpt-4o-mini",
                 summary_model: str = "gpt-3.5-turbo",
                 user_id: str = None,  # Default user_id from settings
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 overlap: int = 50):
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
            user_id: User ID (defaults to settings.DEFAULT_USER_ID)
            max_chunk_size: Maximum size of chunks in characters
            min_chunk_size: Minimum size of chunks in characters
            overlap: Amount of overlap between chunks in characters
        """
        self.storage_provider = storage_provider
        self.vector_db_provider = vector_db_provider
        self.rel_db_provider = rel_db_provider
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.user_id = user_id or settings.DEFAULT_USER_ID
        
        # For MVP, warn if user_id is not the default since we'll use DEFAULT_USER_ID for vector DB
        if self.user_id != settings.DEFAULT_USER_ID:
            logger.warning(f"Using non-default user_id '{user_id}' for storage paths, but vector DB operations will use tenant_id='{settings.DEFAULT_USER_ID}' for MVP")
        
        # Ensure default tenant exists in vector database
        try:
            self._ensure_tenant_exists(settings.DEFAULT_USER_ID)
        except Exception as e:
            logger.warning(f"Could not verify tenant '{settings.DEFAULT_USER_ID}' during initialization. Will try again when processing: {e}")
        
        # Initialize component classes
        self.content_extractor = ContentExtractor()
        self.chunker = Chunker(
            max_chunk_size=max_chunk_size, 
            min_chunk_size=min_chunk_size, 
            overlap=overlap
        )
        
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
        
        # Get DB session - we'll use a single session for the entire operation
        db_session = self.rel_db_provider.get_db_session()
        
        try:
            # Create document record in the database first if it doesn't exist
            s3_path = f"tenant/{self.user_id}/raw/{document_id}_{file_info.filename}"
            document = db_session.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if not document:
                logger.info(f"Creating document record in database for {document_id}")
                document = self.rel_db_provider.create_document(
                    session=db_session,
                    user_id=self.user_id,
                    filename=file_info.filename,
                    content_type=file_info.content_type,
                    s3_path=s3_path,
                    document_id=document_id  # Pass the document ID explicitly
                )
                # Double-check the document was created
                if document:
                    logger.info(f"Document {document_id} created successfully")
                    
                    # Verification step to help debug transaction issues
                    verification = db_session.query(Document).filter(Document.id == document_id).first()
                    if verification:
                        logger.info(f"Verified document {document_id} exists in current session")
                    else:
                        logger.warning(f"Document {document_id} does not exist in current session - possible transaction issue")
                else:
                    logger.error(f"Failed to create document {document_id} - database returned None")
                    return ProcessingResult.failed(document_id, "Failed to create document record")
            else:
                logger.info(f"Document {document_id} already exists in database")
            
            # Update document status to "processing" in the database
            # Pass our existing session for all database operations
            document = self._update_document_status_with_session(db_session, document_id, ProcessingStatus.PROCESSING)
            if not document:
                logger.error(f"Could not update document {document_id} status to PROCESSING")
                return ProcessingResult.failure(document_id, "Could not update document status")
               
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
            
            # Store both insight and summary in Wasabi + PostgreSQL
            # Use our current session for storing insight & summary
            logger.info(f"Storing document insight and summary")
            insight_path = self._store_insight_with_session(db_session, document_id, insight)
            summary_path = self._store_summary_with_session(db_session, document_id, summary)
            
            # Step 4: Chunk the content
            logger.info(f"Chunking document content: {file_info.filename}")
            chunks = self._chunk_content(extracted_content, document_id)
            
            # Step 5: Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Step 6: Store chunks and their embeddings in Wasabi + Weaviate
            logger.info(f"Storing {len(chunks_with_embeddings)} chunks and embeddings")
            stored_chunks = self._store_chunks_and_embeddings(chunks_with_embeddings, file_info)
            
            # Step 7: Update document status to "completed" in the database
            document = self._update_document_status_with_session(
                db_session, 
                document_id, 
                ProcessingStatus.COMPLETED, 
                len(stored_chunks)
            )
            
            if not document:
                logger.error(f"Could not update document {document_id} status to COMPLETED")
                return ProcessingResult.failed(document_id, "Could not update document status to COMPLETED")
            
            # Final verification before returning success
            final_check = db_session.query(Document).filter(Document.id == document_id).first()
            if not final_check:
                logger.critical(f"Document {document_id} not found in database at end of processing. This should never happen.")
                return ProcessingResult.failed(document_id, "Document not found in database at end of processing")
                
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
                self._update_document_status_with_session(
                    db_session, 
                    document_id, 
                    ProcessingStatus.FAILED,
                    error_message=str(e)
                )
                db_session.commit()
            except Exception as inner_e:
                logger.error(f"Failed to update document status to failed: {str(inner_e)}")
                
            return ProcessingResult.failure(document_id, str(e))
        finally:
            # Close the database session
            self.rel_db_provider.close_db_session(db_session)
    
    def _update_document_status_with_session(self, 
                               db_session, 
                               document_id: str, 
                               status: ProcessingStatus,
                               chunk_count: int = 0,
                               error_message: Optional[str] = None) -> Optional[Document]:
        """
        Update the document processing status in the database using a provided session.
        
        Args:
            db_session: Database session
            document_id: Document ID
            status: New processing status
            chunk_count: Number of chunks (for completed documents)
            error_message: Error message (for failed documents)
            
        Returns:
            Updated Document object if successful, None otherwise
        """
        # Get status as string
        status_str = status.value
        logger.info(f"Updating document {document_id} status to {status_str}")
        
        # Verify document exists first
        document = db_session.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.warning(f"Cannot update status: Document {document_id} not found in database")
            
            # Create document record if it doesn't exist
            try:
                logger.info(f"Creating document record for {document_id} with status {status_str}")
                s3_path = f"tenant/{self.user_id}/raw/{document_id}"
                document = Document(
                    id=document_id,  # Make sure ID is explicitly set
                    user_id=self.user_id,
                    filename=f"document_{document_id}",  # Default filename
                    content_type="application/octet-stream",  # Default content type
                    s3_path=s3_path
                )
                db_session.add(document)
                db_session.flush()  # Flush to get database errors but don't commit yet
                
                # Verify document was added to session
                verification = db_session.query(Document).filter(Document.id == document_id).first()
                if verification:
                    logger.info(f"Verified document {document_id} exists in session after creation in _update_document_status_with_session")
                else:
                    logger.error(f"Document {document_id} NOT found in session after creation in _update_document_status_with_session")
                    return None
                    
                logger.info(f"Created document record for {document_id}")
            except Exception as e:
                # Add more context to help debug database issues
                logger.error(f"Failed to create document {document_id} in database: {str(e)}")
                try:
                    # Try to get more information about the connection state
                    engine_status = "Unknown"
                    if hasattr(db_session, 'bind') and hasattr(db_session.bind, 'pool'):
                        engine_status = f"Pool size: {db_session.bind.pool.size}, Overflow: {db_session.bind.pool.overflow}, Checkedin: {db_session.bind.pool.checkedin()}"
                    logger.error(f"Database engine status: {engine_status}")
                except:
                    pass
                return None
        
        # Now update the status
        try:
            if status == ProcessingStatus.COMPLETED:
                # Update document as processed with chunk count
                document.processed = True
                document.chunk_count = chunk_count
                db_session.flush()  # Flush changes but don't commit yet
                logger.info(f"Document {document_id} marked as processed with {chunk_count} chunks")
            elif status == ProcessingStatus.FAILED:
                # Mark as not processed
                document.processed = False
                document.chunk_count = 0
                db_session.flush()  # Flush changes but don't commit yet
                logger.info(f"Document {document_id} marked as failed: {error_message}")
            
            return document
        except Exception as e:
            logger.error(f"Error updating document {document_id} status: {str(e)}")
            return None
            
    def _store_insight_with_session(self, db_session, document_id: str, insight: DocumentInsight) -> str:
        """
        Store document insight in the storage provider and relational database using provided session.
        
        Args:
            db_session: Database session
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
        
        # Store in relational database using the provided session
        try:
            document = db_session.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if document:
                logger.info(f"Updating document {document_id} with insight")
                document.insight = insight_dict
                document.title = insight.title
                db_session.flush()  # Flush changes but don't commit yet
            else:
                # Document should already exist by this point in the pipeline
                logger.critical(f"Document {document_id} not found in database when storing insight. This should never happen.")
        except Exception as e:
            logger.error(f"Error storing insight in database: {str(e)}")
        
        return s3_uri
    
    def _store_summary_with_session(self, db_session, document_id: str, summary: DocumentSummary) -> str:
        """
        Store document summary in the storage provider and relational database using provided session.
        
        Args:
            db_session: Database session
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
        
        # Store in relational database using the provided session
        try:
            document = db_session.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if document:
                logger.info(f"Updating document {document_id} with summary")
                document.summary = summary_dict
                # Only set title from summary if not already set from insight
                if not document.title:
                    document.title = summary.title
                db_session.flush()  # Flush changes but don't commit yet
            else:
                # Document should already exist by this point in the pipeline
                logger.critical(f"Document {document_id} not found in database when storing summary. This should never happen.")
        except Exception as e:
            logger.error(f"Error storing summary in database: {str(e)}")
        
        return s3_uri
    
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
                    tenant_id = self.user_id  # Use user_id for vector DB
                    self._ensure_tenant_exists(tenant_id)
                    
                    # Add the chunk to vector DB
                    self.vector_db_provider.add_chunk(
                        tenant_id=tenant_id,
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

    def _verify_document_exists(self, document_id: str) -> bool:
        """
        Helper method to verify a document exists in the database.
        Used for debugging transaction issues.
        
        Args:
            document_id: Document ID to check
            
        Returns:
            Boolean indicating whether the document exists
        """
        try:
            db_session = self.rel_db_provider.get_db_session()
            try:
                document = db_session.query(Document).filter(Document.id == document_id).first()
                exists = document is not None
                logger.info(f"Document {document_id} exists check: {exists}")
                return exists
            finally:
                db_session.close()
        except Exception as e:
            logger.error(f"Error checking if document {document_id} exists: {str(e)}")
            return False
