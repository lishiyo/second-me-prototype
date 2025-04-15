#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.providers.rel_db import RelationalDB, Document
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter
from app.providers.l1.postgres_adapter import PostgresAdapter
from app.providers.l1.weaviate_adapter import WeaviateAdapter
from app.providers.vector_db import VectorDB
from app.processors.l0.utils import setup_logger

# Setup logger
logger = setup_logger("test_document_embedding")

def test_document_embedding_pipeline():
    """Test the document embedding pipeline by verifying all methods used in _extract_notes_from_l0"""
    # Initialize PostgreSQL adapter
    rel_db = RelationalDB(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )
    
    # Initialize L1 PostgresAdapter with RelationalDB instance
    postgres_adapter = PostgresAdapter(rel_db=rel_db)
    
    # Get a session for direct DB operations if needed
    session = rel_db.get_db_session()
    
    # Initialize Wasabi adapter
    wasabi_adapter = WasabiStorageAdapter(
        endpoint_url=settings.WASABI_ENDPOINT,
        access_key=settings.WASABI_ACCESS_KEY,
        secret_key=settings.WASABI_SECRET_KEY,
        bucket_name=settings.WASABI_BUCKET,
        region_name=settings.WASABI_REGION
    )
    
    # Initialize Vector DB for the weaviate adapter
    vector_db = VectorDB(
        url=settings.WEAVIATE_URL,
        api_key=settings.WEAVIATE_API_KEY,
        embedding_model=settings.EMBEDDING_MODEL
    )
    
    # Initialize Weaviate adapter
    weaviate_adapter = WeaviateAdapter(client=vector_db.client)
    
    # Use the default user ID for testing
    user_id = settings.DEFAULT_USER_ID
    
    # 1. Test get_documents_with_l0
    logger.info("Testing get_documents_with_l0...")
    documents = postgres_adapter.get_documents_with_l0(user_id)
    logger.info(f"Found {len(documents)} documents with L0 data for user {user_id}")
    
    if not documents:
        logger.error("No documents found! Please ensure documents are processed.")
        return False
    
    # Process each document
    success_count = 0
    for doc in documents:
        doc_id = doc.id
        logger.info(f"\nTesting document: {doc_id} - {doc.title or 'No title'}")
        
        # 2. Test get_document_embedding
        logger.info("Testing weaviate_adapter.get_document_embedding...")
        doc_embedding = weaviate_adapter.get_document_embedding(user_id, doc_id)
        
        if doc_embedding:
            logger.info(f"✅ Document embedding found: length={len(doc_embedding)}")
        else:
            logger.error(f"❌ Document embedding NOT found for {doc_id}")
            continue
        
        # 3. Test get_document_chunks
        logger.info("Testing weaviate_adapter.get_document_chunks...")
        chunks = weaviate_adapter.get_document_chunks(user_id, doc_id)
        
        if chunks and len(chunks) > 0:
            logger.info(f"✅ Document chunks found: count={len(chunks)}")
        else:
            logger.error(f"❌ Document chunks NOT found for {doc_id}")
            continue
        
        # 4. Test get_chunk_embeddings_by_document
        logger.info("Testing weaviate_adapter.get_chunk_embeddings_by_document...")
        chunk_embeddings = weaviate_adapter.get_chunk_embeddings_by_document(user_id, doc_id)
        
        if chunk_embeddings and len(chunk_embeddings) > 0:
            logger.info(f"✅ Chunk embeddings found: count={len(chunk_embeddings)}")
        else:
            logger.error(f"❌ Chunk embeddings NOT found for {doc_id}")
            continue
        
        # 5. Test get_document from Wasabi
        logger.info("Testing Wasabi.get_document...")
        try:
            document_data = wasabi_adapter.get_document(user_id, doc_id)
            if document_data:
                logger.info(f"✅ Document data found in Wasabi")
                insight_data = document_data.get("insight", {}) or {}
                summary_data = document_data.get("summary", {}) or {}
                
                if insight_data:
                    logger.info(f"✅ Insight found: {insight_data.get('title', 'No title')}")
                if summary_data:
                    logger.info(f"✅ Summary found: {len(summary_data.get('summary', '')) > 0}")
            else:
                logger.warning(f"⚠️ Document data NOT found in Wasabi for {doc_id}")
        except Exception as e:
            logger.error(f"❌ Error getting document data from Wasabi: {e}")
            continue
        
        # All tests passed for this document
        success_count += 1
        logger.info(f"✅ All tests passed for document {doc_id}")
    
    # Close session
    rel_db.close_db_session(session)
    
    # Overall results
    if success_count == 0:
        logger.error("❌ No documents passed all tests!")
        return False
    elif success_count < len(documents):
        logger.warning(f"⚠️ {success_count}/{len(documents)} documents passed all tests")
        return True
    else:
        logger.info(f"✅ All {success_count} documents passed all tests!")
        return True

if __name__ == "__main__":
    success = test_document_embedding_pipeline()
    if success:
        logger.info("Document embedding verification completed successfully!")
    else:
        logger.error("Document embedding verification failed!")
        sys.exit(1) 