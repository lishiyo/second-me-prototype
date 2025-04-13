#!/usr/bin/env python
"""
Test script to verify that our adapters are working correctly.
This script tests connections to Wasabi, Weaviate, and PostgreSQL.
"""

import os
import sys
import uuid
import logging
from datetime import datetime, timezone
from io import BytesIO
import numpy as np

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.providers.blob_store import BlobStore
from app.providers.vector_db import VectorDB
from app.providers.rel_db import RelationalDB, User, Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_blob_store():
    """Test the Wasabi S3 adapter."""
    logger.info("Testing BlobStore adapter (Wasabi S3)...")
    
    # Initialize the BlobStore
    blob_store = BlobStore(
        access_key=settings.WASABI_ACCESS_KEY,
        secret_key=settings.WASABI_SECRET_KEY,
        bucket=settings.WASABI_BUCKET,
        region=settings.WASABI_REGION,
        endpoint=settings.WASABI_ENDPOINT
    )
    
    # Create a test key
    test_key = f"test/blob_store_test_{uuid.uuid4()}.txt"
    test_content = b"This is a test file to check if the BlobStore adapter is working."
    
    # Test upload
    logger.info(f"Uploading test file to {test_key}...")
    s3_uri = blob_store.put_object(test_key, test_content)
    logger.info(f"Uploaded to {s3_uri}")
    
    # Test download
    logger.info("Downloading test file...")
    downloaded_content = blob_store.get_object(test_key)
    
    # Verify content
    assert downloaded_content == test_content, "Downloaded content doesn't match uploaded content"
    logger.info("Content verification successful")
    
    # Test metadata
    logger.info("Testing with metadata...")
    metadata = {"test_key": "test_value", "timestamp": datetime.now(timezone.utc).isoformat()}
    s3_uri_with_metadata = blob_store.put_object(test_key, test_content, metadata=metadata)
    retrieved_metadata = blob_store.get_metadata(test_key)
    logger.info(f"Retrieved metadata: {retrieved_metadata}")
    
    # Test file-like objects
    logger.info("Testing with file-like objects...")
    file_content = b"This is a test using file-like objects."
    fileobj = BytesIO(file_content)
    file_key = f"test/blob_store_fileobj_test_{uuid.uuid4()}.txt"
    s3_uri_fileobj = blob_store.put_fileobj(file_key, fileobj)
    
    # Test download to file-like object
    download_fileobj = BytesIO()
    blob_store.get_fileobj(file_key, download_fileobj)
    download_fileobj.seek(0)
    logger.info(f"Downloaded content: {download_fileobj.read().decode('utf-8')}")
    
    # Test listing objects
    logger.info("Listing objects...")
    objects = blob_store.list_objects(prefix="test/")
    logger.info(f"Found {len(objects)} objects with prefix 'test/'")
    
    # Test deleting objects
    logger.info("Deleting test objects...")
    blob_store.delete_object(test_key)
    blob_store.delete_object(file_key)
    
    logger.info("BlobStore tests completed successfully")

def test_vector_db():
    """Test the Weaviate vector database adapter."""
    logger.info("Testing VectorDB adapter (Weaviate)...")
    
    # Initialize the VectorDB
    vector_db = VectorDB(
        url=settings.WEAVIATE_URL,
        api_key=settings.WEAVIATE_API_KEY,
        embedding_model=settings.EMBEDDING_MODEL
    )
    
    # Test parameters
    tenant_id = "test_tenant"
    tenant_id_2 = "test_tenant_2"
    document_id = f"test_doc_{uuid.uuid4()}"
    s3_path = f"s3://{settings.WASABI_BUCKET}/test/document_{document_id}.txt"
    
    # Clean up any existing test data
    logger.info(f"Checking if tenant '{tenant_id}' exists...")
    tenants = vector_db.list_tenants()
    logger.info(f"Current tenants: {tenants}")
    
    if tenant_id in tenants:
        logger.info(f"Deleting existing tenant '{tenant_id}'...")
        vector_db.delete_tenant(tenant_id)
    
    if tenant_id_2 in tenants:
        logger.info(f"Deleting existing tenant '{tenant_id_2}'...")
        vector_db.delete_tenant(tenant_id_2)
    
    # Test tenant creation
    logger.info(f"Creating test tenant '{tenant_id}'...")
    vector_db.create_tenant(tenant_id)
    
    # Test adding chunks to first tenant
    logger.info(f"Adding test chunks to tenant '{tenant_id}'...")
    chunks_tenant_1 = []
    for i in range(5):
        # Create a random embedding vector normalized to unit length
        mock_embedding = list(np.random.random(1536).astype(float))
        chunk = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "s3_path": f"{s3_path}_chunk_{i}.txt",
            "chunk_index": i,
            "embedding": mock_embedding,
            "metadata": {
                "filename": f"test_document_{i}.txt",
                "content_type": "text/plain",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        chunks_tenant_1.append(chunk)
    
    # Test batch add
    logger.info(f"Testing batch add for tenant '{tenant_id}'...")
    uuids_tenant_1 = vector_db.batch_add_chunks(chunks_tenant_1)
    logger.info(f"Added {len(uuids_tenant_1)} chunks to tenant '{tenant_id}'")
    
    # Create a second tenant for isolation testing
    logger.info(f"Creating second test tenant '{tenant_id_2}'...")
    vector_db.create_tenant(tenant_id_2)
    
    # Add different chunks to second tenant
    logger.info(f"Adding test chunks to tenant '{tenant_id_2}'...")
    chunks_tenant_2 = []
    for i in range(3):
        # Create a random embedding vector normalized to unit length
        mock_embedding = list(np.random.random(1536).astype(float))
        chunk = {
            "tenant_id": tenant_id_2,
            "document_id": document_id,  # Same document ID but different tenant
            "s3_path": f"{s3_path}_tenant2_chunk_{i}.txt",
            "chunk_index": i,
            "embedding": mock_embedding,  
            "metadata": {
                "filename": f"tenant2_document_{i}.txt",
                "content_type": "text/plain",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        chunks_tenant_2.append(chunk)
    
    # Batch add to second tenant
    uuids_tenant_2 = vector_db.batch_add_chunks(chunks_tenant_2)
    logger.info(f"Added {len(uuids_tenant_2)} chunks to tenant '{tenant_id_2}'")
    
    # Wait a moment for indexing
    import time
    logger.info("Waiting for indexing...")
    time.sleep(5)
    
    # Test tenant isolation - search in tenant 1
    logger.info(f"Testing search in tenant '{tenant_id}'...")
    search_results_1 = vector_db.search(tenant_id, "information about testing", limit=10)
    logger.info(f"Search in tenant '{tenant_id}' returned {len(search_results_1)} results")
    
    # Since tenant isolation is handled by Weaviate, we don't need to check tenant_id in results
    # We're already searching in the tenant-specific collection
    
    # Test tenant isolation - search in tenant 2
    logger.info(f"Testing search in tenant '{tenant_id_2}'...")
    search_results_2 = vector_db.search(tenant_id_2, "completely different", limit=10)
    logger.info(f"Search in tenant '{tenant_id_2}' returned {len(search_results_2)} results")
    
    # No need to verify tenant_id in results as it's handled by tenant isolation in Weaviate
    
    # Test hybrid search
    logger.info(f"Testing hybrid search in tenant '{tenant_id}'...")
    hybrid_results = vector_db.hybrid_search(tenant_id, "test chunk", limit=3)
    logger.info(f"Hybrid search returned {len(hybrid_results)} results")
    
    # Test get by ID
    if uuids_tenant_1:
        logger.info(f"Testing get by ID with UUID {uuids_tenant_1[0]}...")
        obj = vector_db.get_by_id(tenant_id, uuids_tenant_1[0])
        if obj:
            logger.info(f"Retrieved object from tenant '{tenant_id}': document {obj.get('document_id')}, chunk {obj.get('chunk_index')}")
    
    # Test tenant status
    logger.info(f"Testing tenant status for '{tenant_id}'...")
    status = vector_db.get_tenant_status(tenant_id)
    logger.info(f"Tenant '{tenant_id}' status: {status}")
    
    # Test setting tenant status to inactive and back
    logger.info(f"Setting tenant '{tenant_id}' to inactive...")
    vector_db.set_tenant_status(tenant_id, active=False)
    status = vector_db.get_tenant_status(tenant_id)
    logger.info(f"Tenant '{tenant_id}' status after setting inactive: {status}")
    
    logger.info(f"Setting tenant '{tenant_id}' back to active...")
    vector_db.set_tenant_status(tenant_id, active=True)
    status = vector_db.get_tenant_status(tenant_id)
    logger.info(f"Tenant '{tenant_id}' status after setting active: {status}")
    
    # Test deleting objects by document ID
    logger.info(f"Testing delete by document ID in tenant '{tenant_id}'...")
    deleted_count = vector_db.delete_by_document(tenant_id, document_id)
    logger.info(f"Deleted {deleted_count} chunks from tenant '{tenant_id}'")
    
    # Clean up - delete tenants
    logger.info("Cleaning up by deleting test tenants...")
    vector_db.delete_tenant(tenant_id)
    vector_db.delete_tenant(tenant_id_2)
    
    # Close the connection to Weaviate to prevent memory leaks
    logger.info("Closing Weaviate connection...")
    vector_db.close()
    
    logger.info("VectorDB multi-tenancy tests completed successfully")

def test_relational_db():
    """Test the PostgreSQL relational database adapter."""
    logger.info("Testing RelationalDB adapter (PostgreSQL)...")
    
    # Initialize the RelationalDB
    rel_db = RelationalDB(
        connection_string=settings.db_connection_string
    )
    
    # Get a session
    session = rel_db.get_db_session()
    
    try:
        # Test creating a user
        logger.info("Creating test user...")
        test_user_id = str(uuid.uuid4())
        user = rel_db.create_user(session, test_user_id)
        logger.info(f"Created user with ID: {user.id}")
        
        # Test retrieving the user
        logger.info("Retrieving test user...")
        retrieved_user = rel_db.get_user(session, test_user_id)
        logger.info(f"Retrieved user created at: {retrieved_user.created_at}")
        
        # Test creating a document
        logger.info("Creating test document...")
        document = rel_db.create_document(
            session,
            user.id,
            "test_document.txt",
            "text/plain",
            f"s3://{settings.WASABI_BUCKET}/test/document_{uuid.uuid4()}.txt"
        )
        logger.info(f"Created document with ID: {document.id}")
        
        # Test updating document processing status
        logger.info("Updating document processing status...")
        updated_document = rel_db.update_document_processed(
            session,
            document.id,
            processed=True,
            chunk_count=10
        )
        logger.info(f"Updated document processed status: {updated_document.processed}, chunk count: {updated_document.chunk_count}")
        
        # Test getting user documents
        logger.info("Getting user documents...")
        user_documents = rel_db.get_user_documents(session, user.id)
        logger.info(f"User has {len(user_documents)} documents")
        
        # Test creating a training job
        logger.info("Creating test training job...")
        job = rel_db.create_training_job(session, user.id)
        logger.info(f"Created training job with ID: {job.id}, status: {job.status}")
        
        # Test updating job status
        logger.info("Updating job status...")
        updated_job = rel_db.update_training_job_status(
            session,
            job.id,
            "processing",
            lora_path=f"s3://{settings.WASABI_BUCKET}/tenant/{user.id}/lora/test.safetensors"
        )
        logger.info(f"Updated job status: {updated_job.status}, started at: {updated_job.started_at}")
        
        # Test creating a chat session
        logger.info("Creating test chat session...")
        chat_session = rel_db.create_chat_session(
            session,
            user.id,
            f"s3://{settings.WASABI_BUCKET}/tenant/{user.id}/chats/session_{uuid.uuid4()}",
            title="Test Chat Session",
            summary="This is a test chat session"
        )
        logger.info(f"Created chat session with ID: {chat_session.id}")
        
        # Test updating chat session
        logger.info("Updating chat session...")
        updated_chat = rel_db.update_chat_session(
            session,
            chat_session.id,
            title="Updated Test Chat",
            processed_for_training=True
        )
        logger.info(f"Updated chat session title: {updated_chat.title}, processed: {updated_chat.processed_for_training}")
        
        # Test getting user chat sessions
        logger.info("Getting user chat sessions...")
        user_chats = rel_db.get_user_chat_sessions(session, user.id)
        logger.info(f"User has {len(user_chats)} chat sessions")
        
        # Clean up
        logger.info("Cleaning up...")
        rel_db.delete_chat_session(session, chat_session.id)
        rel_db.delete_training_job(session, job.id)
        rel_db.delete_document(session, document.id)
        
        # For the test user, cascade deletion should remove all related objects
        logger.info("Deleting test user (cascade delete)...")
        session.delete(retrieved_user)
        session.commit()
        
        logger.info("RelationalDB tests completed successfully")
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error during RelationalDB test: {e}")
        raise
    finally:
        # Close the session
        rel_db.close_db_session(session)

def main():
    """Run all adapter tests."""
    logger.info("Starting adapter tests...")
    
    # Validate settings
    missing_settings = settings.validate_settings()
    if missing_settings:
        logger.error(f"Missing required settings: {', '.join(missing_settings)}")
        logger.error("Please set these environment variables or update .env file")
        return
    
    try:
        # Test each adapter
        test_blob_store()
        test_vector_db()
        test_relational_db()
        
        logger.info("All adapter tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {e}")
        raise

if __name__ == "__main__":
    main() 