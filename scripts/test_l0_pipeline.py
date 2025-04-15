#!/usr/bin/env python3
"""
Test script for L0 processing pipeline integration.
This script demonstrates the complete L0 processing flow with a real document.
This uses the two-step processor that first generates an insight and then a summary.

Usage:
    python scripts/test_l0_pipeline.py [file_path]
    
If file_path is not provided, it defaults to 'data/25 items.md'.
"""

import os
import sys
import uuid
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).joinpath('.env')
load_dotenv(dotenv_path=dotenv_path)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.processors.l0.models import FileInfo, ProcessingStatus
from app.processors.l0.document_processor import DocumentProcessor
from app.providers.blob_store import BlobStore
from app.providers.vector_db import VectorDB
from app.providers.rel_db import RelationalDB
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing configuration
)

# Prevent propagation from app loggers to avoid duplicate logs
for logger_name in logging.Logger.manager.loggerDict:
    if logger_name.startswith('app.'):
        logging.getLogger(logger_name).propagate = False

logger = logging.getLogger(__name__)

def init_storage_providers():
    """Initialize all storage providers needed for the pipeline."""
    # Initialize BlobStore (Wasabi)
    # For testing, we'll use local filesystem or MinIO if available
    blob_store = BlobStore(
        access_key=settings.WASABI_ACCESS_KEY,
        secret_key=settings.WASABI_SECRET_KEY,
        bucket=settings.WASABI_BUCKET,
        region=settings.WASABI_REGION,
        endpoint=settings.WASABI_ENDPOINT
    )
    
    # Initialize VectorDB (Weaviate)
    vector_db = VectorDB(
        url=settings.WEAVIATE_URL,
        api_key=settings.WEAVIATE_API_KEY,
        embedding_model=settings.EMBEDDING_MODEL
    )
    
    # Initialize RelationalDB (PostgreSQL)
    rel_db = RelationalDB(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )
    
    return blob_store, vector_db, rel_db

def process_file(file_path, document_processor):
    """Process a file through the L0 pipeline."""
    # Generate a document ID
    document_id = str(uuid.uuid4())
    
    # Determine content type based on file extension
    _, extension = os.path.splitext(file_path)
    content_type = {
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.html': 'text/html',
    }.get(extension.lower(), 'application/octet-stream')
    
    # Get filename from path
    filename = os.path.basename(file_path)
    
    # Read file content
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Create FileInfo object
    file_info = FileInfo(
        document_id=document_id,
        filename=filename,
        content_type=content_type,
        s3_path=file_path,  # Local path for reference
        content=content     # Actual content
    )
    
    # Process the document - document creation now happens inside process_document
    result = document_processor.process_document(file_info)
    
    # Return the processing result
    return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the L0 processing pipeline with a document.')
    parser.add_argument('file_path', nargs='?', default=None, 
                        help='Path to the file to process. Defaults to "data/25 items.md".')
    return parser.parse_args()

def main():
    """Main function to run the test."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check for OpenAI API key
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable is required")
        logger.error("Make sure you have a .env file in the project root with OPENAI_API_KEY=your_key")
        logger.error("Or set it directly in your environment with: export OPENAI_API_KEY=your_key")
        sys.exit(1)
    
    # Initialize storage providers
    try:
        blob_store, vector_db, rel_db = init_storage_providers()
        logger.info("Storage providers initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing storage providers: {e}")
        sys.exit(1)
    
    try:
        # Use the default user ID from settings
        user_id = settings.DEFAULT_USER_ID
        logger.info(f"Using user_id: {user_id}")
        
        # Get or create user in database
        db_session = rel_db.get_db_session()
        try:
            user = rel_db.get_or_create_user(db_session, user_id=user_id)
            logger.info(f"Using user with ID: {user.id}")
        except Exception as e:
            logger.error(f"Error getting or creating user: {e}")
            db_session.rollback()
            raise
        finally:
            rel_db.close_db_session(db_session)
        
        # Create document processor with chunking settings from config
        document_processor = DocumentProcessor(
            storage_provider=blob_store,
            vector_db_provider=vector_db,
            rel_db_provider=rel_db,
            openai_api_key=settings.OPENAI_API_KEY,
            chunking_strategy="paragraph",
            user_id=user_id,
            # Use chunking settings from config
            max_chunk_size=settings.CHUNK_SIZE,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
            overlap=settings.OVERLAP
        )
        
        # Define the file to process
        default_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '25 items.md'))
        file_path = args.file_path if args.file_path else default_file_path
        
        # Validate file existence
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        
        logger.info(f"Processing file: {file_path}")
        
        # Process the file
        try:
            result = process_file(file_path, document_processor)
            
            # Print results
            logger.info("Processing completed with results:")
            logger.info(f"Status: {result.status.value}")
            logger.info(f"Document ID: {result.document_id}")
            logger.info(f"Chunk count: {result.chunk_count}")
            
            # Print insight information
            if result.insight:
                logger.info(f"Title: {result.insight.title}")
                logger.info(f"Insight: {result.insight.insight}")
            
            # Print summary information
            if result.summary:
                logger.info(f"Title: {result.summary.title}")
                logger.info(f"Summary: {result.summary.summary}")
                logger.info(f"Keywords: {', '.join(result.summary.keywords)}")
            
            if result.status == ProcessingStatus.FAILED:
                logger.error(f"Error: {result.error}")
            
            # Verify document embedding was created
            if result.status == ProcessingStatus.COMPLETED:
                try:
                    doc_embedding = vector_db.get_document_embedding(user_id, result.document_id)
                    if doc_embedding:
                        logger.info(f"✅ Document embedding created successfully (dim: {len(doc_embedding['embedding'])})")
                    else:
                        logger.warning("❌ Document embedding not found")
                except Exception as e:
                    logger.error(f"Error verifying document embedding: {e}")
            
            # Success!
            logger.info("Test completed successfully")
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    finally:
        # Clean up connections
        logger.info("Cleaning up connections...")
        vector_db.close()  # Close the Weaviate client connection

if __name__ == "__main__":
    main() 