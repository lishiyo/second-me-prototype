#!/usr/bin/env python3
"""
Process all files in the data directory through the L0 pipeline.
This script will find all files in the data directory and process them
through the L0 pipeline, storing results in all backend systems.

Usage:
    python scripts/process_all_data.py [--skip-existing]
    
Optional arguments:
    --skip-existing: Skip files that have already been processed
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).joinpath('.env')
load_dotenv(dotenv_path=dotenv_path)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.processors.l0.models import FileInfo
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
    """Process a single file through the L0 pipeline."""
    import uuid
    
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
    
    # Process the document
    result = document_processor.process_document(file_info)
    
    # Return the processing result
    return result

def has_been_processed(filename, rel_db):
    """Check if a file has already been processed."""
    db_session = rel_db.get_db_session()
    try:
        # Get all documents to check if the filename exists
        documents = rel_db.get_user_documents(db_session, settings.DEFAULT_USER_ID)
        for doc in documents:
            if doc.filename == filename and doc.processed:
                return True
        return False
    finally:
        rel_db.close_db_session(db_session)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process all files in data directory')
    parser.add_argument('--skip-existing', action='store_true', 
                        help='Skip files that have already been processed')
    return parser.parse_args()

def main():
    """Main function to process all files."""
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
        
        # Create document processor
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
        
        # Get data directory path
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        logger.info(f"Scanning directory: {data_dir}")
        
        # Get all files in the data directory
        file_list = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
        
        logger.info(f"Found {len(file_list)} files to process")
        
        # Process each file
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for i, file_path in enumerate(file_list):
            filename = os.path.basename(file_path)
            logger.info(f"[{i+1}/{len(file_list)}] Processing file: {filename}")
            
            # Check if file has been processed already
            if args.skip_existing and has_been_processed(filename, rel_db):
                logger.info(f"Skipping already processed file: {filename}")
                skip_count += 1
                continue
            
            try:
                # Process the file
                result = process_file(file_path, document_processor)
                
                # Log results
                logger.info(f"Status: {result.status.value}")
                logger.info(f"Document ID: {result.document_id}")
                logger.info(f"Chunk count: {result.chunk_count}")
                
                # Print insight/summary information if available
                if result.insight:
                    logger.info(f"Title: {result.insight.title}")
                    # Truncate insight for log readability
                    insight_preview = result.insight.insight[:100] + "..." if len(result.insight.insight) > 100 else result.insight.insight
                    logger.info(f"Insight: {insight_preview}")
                
                if result.summary:
                    logger.info(f"Keywords: {', '.join(result.summary.keywords)}")
                
                if result.status.value == "FAILED":
                    logger.error(f"Error: {result.error}")
                    error_count += 1
                else:
                    success_count += 1
                    
                # Add a divider for readability
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                error_count += 1
                # Continue with next file
                continue
        
        # Report final statistics
        logger.info("=" * 50)
        logger.info("Processing complete!")
        logger.info(f"Total files processed: {len(file_list)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Skipped: {skip_count}")
        logger.info(f"Failed: {error_count}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        sys.exit(1)
    finally:
        # Clean up connections
        logger.info("Cleaning up connections...")
        try:
            vector_db.close()  # Close the Weaviate client connection
        except:
            pass

if __name__ == "__main__":
    main() 