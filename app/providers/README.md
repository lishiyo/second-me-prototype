# Core Service Adapters

This directory contains adapter classes for interacting with various backend services used in the Second Me Digital Twin application.

## Blob Store Adapter (`blob_store.py`)

A wrapper around AWS S3 SDK (boto3) configured to work with Wasabi S3-compatible storage. This adapter handles all file operations, including:

- Uploading and downloading files and binary objects
- Managing file metadata
- Listing and deleting objects
- Generating pre-signed URLs for temporary access

The storage is organized as follows:
```
tenant/<user_id>/
  ├── raw/                    # Original documents
  ├── chunks/                 # Processed document chunks
  ├── chats/                  # Chat session files
  ├── training_data/          # Processed training data
  ├── metadata/               # GraphRAG structure
  └── lora/                   # Training artifacts
```

### Key Methods:

- `put_object()` / `put_file()` / `put_fileobj()`: Upload data to S3
- `get_object()` / `get_fileobj()`: Download data from S3
- `list_objects()`: List objects with a given prefix
- `delete_object()`: Delete an object from S3
- `get_metadata()`: Retrieve object metadata

## Vector Database Adapter (`vector_db.py`)

A wrapper around the Weaviate client for storing and searching vector embeddings. This adapter implements multi-tenancy support to ensure data isolation between users.

The adapter creates a `TenantChunk` collection with multi-tenancy enabled, ensuring that:
- Each tenant's data is stored in a separate shard
- Data from one tenant is not visible to other tenants
- Queries are scoped to a specific tenant

### Key Methods:

- `add_chunk()` / `batch_add_chunks()`: Add document chunks to the vector store
- `search()`: Perform semantic vector search within a tenant's data
- `hybrid_search()`: Perform hybrid (vector + keyword) search within a tenant's data
- `delete_by_document()`: Delete all chunks for a document
- `create_tenant()` / `delete_tenant()`: Manage tenants
- `get_tenant_status()` / `set_tenant_status()`: Control tenant activity status

## Relational Database Adapter (`rel_db.py`)

A wrapper around SQLAlchemy for working with PostgreSQL. This adapter manages structured data models for:

- Users
- Document metadata
- Training job status
- Chat session metadata

The adapter provides session management and CRUD operations for all models with proper error handling.

### Key Methods:

- Database operations:
  - `get_db_session()` / `close_db_session()`: Manage database sessions

- User operations:
  - `create_user()` / `get_user()`

- Document operations:
  - `create_document()` / `get_document()` / `get_user_documents()`
  - `update_document_processed()` / `delete_document()`

- Training job operations:
  - `create_training_job()` / `get_training_job()` / `get_user_training_jobs()`
  - `update_training_job_status()` / `delete_training_job()`

- Chat session operations:
  - `create_chat_session()` / `get_chat_session()` / `get_user_chat_sessions()`
  - `update_chat_session()` / `delete_chat_session()`

## Usage Examples

### Blob Store

```python
from app.core.dependencies import get_blob_store

# Get the blob store instance
blob_store = get_blob_store()

# Upload a document
user_id = "1"
document_key = f"tenant/{user_id}/raw/document.pdf"
with open("local_document.pdf", "rb") as f:
    s3_uri = blob_store.put_fileobj(document_key, f)

# Download a document
content = blob_store.get_object(document_key)
```

### Vector Database

```python
from app.core.dependencies import get_vector_db

# Get the vector database instance
vector_db = get_vector_db()

# Add a document chunk
user_id = "1"
chunk_id = vector_db.add_chunk(
    tenant_id=user_id,
    document_id="doc123",
    s3_path="s3://bucket/tenant/1/chunks/doc123/chunk_0.txt",
    chunk_index=0,
    content="This is a sample chunk of text for embeddings",
    metadata={"filename": "document.pdf", "content_type": "application/pdf"}
)

# Search for similar chunks
results = vector_db.search(
    tenant_id=user_id,
    query="sample text",
    limit=5
)
```

### Relational Database

```python
from app.core.dependencies import get_relational_db, get_db_session

# Get the database instance
rel_db = get_relational_db()

# Use a session
with get_db_session() as session:
    # Create a document record
    document = rel_db.create_document(
        session=session,
        user_id="1",
        filename="document.pdf",
        content_type="application/pdf",
        s3_path="s3://bucket/tenant/1/raw/document.pdf"
    )
    
    # Update processing status
    rel_db.update_document_processed(
        session=session,
        document_id=document.id,
        processed=True,
        chunk_count=15
    )
``` 