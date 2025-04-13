#!/usr/bin/env python3
"""
Test script for L0 processing pipeline integration.
This script demonstrates the complete L0 processing flow with a real document.

Tested using `data/25 items.md` file.
"""

import os
import sys
import uuid
import logging
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
        access_key=os.environ.get('WASABI_ACCESS_KEY', 'minioadmin'),
        secret_key=os.environ.get('WASABI_SECRET_KEY', 'minioadmin'),
        bucket=os.environ.get('WASABI_BUCKET', 'second-me'),
        region=os.environ.get('WASABI_REGION', 'us-west-1'),
        endpoint=os.environ.get('WASABI_ENDPOINT', 'http://localhost:9000')
    )
    
    # Initialize VectorDB (Weaviate)
    vector_db = VectorDB(
        url=os.environ.get('WEAVIATE_URL', 'http://localhost:8080'),
        api_key=os.environ.get('WEAVIATE_API_KEY', 'weaviate-api-key'),
        embedding_model=os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    )
    
    # Initialize RelationalDB (PostgreSQL)
    rel_db = RelationalDB(
        host=os.environ.get('DB_HOST', 'localhost'),
        port=os.environ.get('DB_PORT', '5432'),
        database=os.environ.get('DB_NAME', 'second_me'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'postgres')
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
    
    # Create document in the database
    db_session = document_processor.rel_db_provider.get_db_session()
    try:
        # Create document record
        document_processor.rel_db_provider.create_document(
            session=db_session,
            user_id=document_processor.user_id,  # UUID from document_processor
            filename=filename,
            content_type=content_type,
            s3_path=f"tenant/{document_processor.user_id}/raw/{document_id}_{filename}"
        )
        db_session.commit()
    except Exception as e:
        logger.error(f"Error creating document record: {e}")
        db_session.rollback()
        raise
    finally:
        document_processor.rel_db_provider.close_db_session(db_session)
    
    # Process the document
    result = document_processor.process_document(file_info)
    
    # Return the processing result
    return result

def main():
    """Main function to run the test."""
    # Check for OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
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
        # Use the default tenant ID "1" for testing
        user_id = DocumentProcessor.DEFAULT_TENANT_ID
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
        
        # Create document processor
        document_processor = DocumentProcessor(
            storage_provider=blob_store,
            vector_db_provider=vector_db,
            rel_db_provider=rel_db,
            openai_api_key=openai_api_key,
            chunking_strategy="paragraph",
            user_id=user_id  # Use the UUID we created
        )
        
        # Define the file to process
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '25 items.md'))
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
            
            if result.insights:
                logger.info(f"Title: {result.insights.title}")
                logger.info(f"Summary: {result.insights.summary}")
                logger.info(f"Keywords: {', '.join(result.insights.keywords)}")
            
            if result.status == ProcessingStatus.FAILED:
                logger.error(f"Error: {result.error}")
            
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