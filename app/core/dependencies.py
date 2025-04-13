from typing import Generator, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.providers.blob_store import BlobStore
from app.providers.vector_db import VectorDB
from app.providers.rel_db import RelationalDB, User

# Initialize providers for global reuse
_blob_store = None
_vector_db = None
_rel_db = None

def get_blob_store() -> BlobStore:
    """
    Get or create a BlobStore instance.
    
    Returns:
        Configured BlobStore instance
    """
    global _blob_store
    if _blob_store is None:
        _blob_store = BlobStore(
            access_key=settings.WASABI_ACCESS_KEY,
            secret_key=settings.WASABI_SECRET_KEY,
            bucket=settings.WASABI_BUCKET,
            region=settings.WASABI_REGION,
            endpoint=settings.WASABI_ENDPOINT
        )
    return _blob_store

def get_vector_db() -> VectorDB:
    """
    Get or create a VectorDB instance.
    
    Returns:
        Configured VectorDB instance
    """
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB(
            url=settings.WEAVIATE_URL,
            api_key=settings.WEAVIATE_API_KEY,
            embedding_model=settings.EMBEDDING_MODEL
        )
    return _vector_db

def get_relational_db() -> RelationalDB:
    """
    Get or create a RelationalDB instance.
    
    Returns:
        Configured RelationalDB instance
    """
    global _rel_db
    if _rel_db is None:
        _rel_db = RelationalDB(
            connection_string=settings.db_connection_string
        )
    return _rel_db

def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session and handle automatic closing.
    
    Returns:
        Database session generator
    """
    db = get_relational_db()
    session = db.get_db_session()
    try:
        yield session
    finally:
        db.close_db_session(session)

def get_or_create_default_user(session: Session) -> User:
    """
    Get or create the default user (for MVP we're only using one user).
    
    Args:
        session: Database session
        
    Returns:
        User instance
    """
    db = get_relational_db()
    user = db.get_user(session, settings.DEFAULT_USER_ID)
    if user is None:
        user = db.create_user(session, settings.DEFAULT_USER_ID)
    return user 