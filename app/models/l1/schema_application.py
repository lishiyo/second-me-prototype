"""
Schema application script for L1.
This module provides functions to apply the L1 schemas to PostgreSQL and Weaviate.
"""

import os
import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.providers.rel_db import RelationalDB
from app.providers.vector_db import VectorDB
from app.models.l1.migrations import get_migration_sql
from app.models.l1.weaviate_schema import get_schema_definitions

logger = logging.getLogger(__name__)

def apply_postgres_schema(rel_db: RelationalDB = None) -> bool:
    """
    Apply the L1 schema to PostgreSQL.
    
    Args:
        rel_db: Optional RelationalDB instance. If None, a new one will be created.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Create RelationalDB instance if not provided
        if rel_db is None:
            rel_db = RelationalDB(
                host=os.environ.get("DB_HOST"),
                port=os.environ.get("DB_PORT"),
                database=os.environ.get("DB_NAME"),
                user=os.environ.get("DB_USER"),
                password=os.environ.get("DB_PASSWORD")
            )
        
        # Get a database session
        session = rel_db.get_db_session()
        
        try:
            # Execute each migration SQL statement
            for sql in get_migration_sql():
                session.execute(text(sql))
            
            # Commit the transaction
            session.commit()
            logger.info("Successfully applied L1 PostgreSQL schema")
            return True
            
        except Exception as e:
            # Rollback for any exception, not just SQLAlchemyError
            session.rollback()
            logger.error(f"Error applying L1 PostgreSQL schema: {e}")
            return False
            
        finally:
            # Close the session
            rel_db.close_db_session(session)
    
    except Exception as e:
        logger.error(f"Error creating database connection: {e}")
        return False

def apply_weaviate_schema(vector_db: VectorDB = None) -> bool:
    """
    Apply the L1 schema to Weaviate.
    
    Args:
        vector_db: Optional VectorDB instance. If None, a new one will be created.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Create VectorDB instance if not provided
        if vector_db is None:
            vector_db = VectorDB(
                url=os.environ.get("WEAVIATE_URL"),
                api_key=os.environ.get("WEAVIATE_API_KEY"),
                embedding_model=os.environ.get("EMBEDDING_MODEL")
            )
        
        schemas = get_schema_definitions()
        success = True
        
        # Check if the collections already exist
        existing_collections = vector_db.client.collections.list_all()
        
        # Create each collection if it doesn't exist
        for schema in schemas:
            collection_name = schema["name"]
            
            if collection_name in existing_collections:
                logger.info(f"Collection {collection_name} already exists, skipping")
                continue
            
            try:
                logger.info(f"Creating Weaviate collection {collection_name}")
                vector_db.client.collections.create(
                    name=collection_name,
                    description=schema["description"],
                    vectorizer_config=schema["vectorizer"],
                    multi_tenancy_config=schema["multi_tenancy"],
                    properties=schema["properties"]
                )
                logger.info(f"Successfully created collection {collection_name}")
            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")
                success = False
        
        return success
    
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return False

def apply_all_schemas() -> bool:
    """
    Apply both PostgreSQL and Weaviate schemas.
    
    Returns:
        bool: True if all successful, False otherwise.
    """
    postgres_success = apply_postgres_schema()
    weaviate_success = apply_weaviate_schema()
    
    return postgres_success and weaviate_success

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if apply_all_schemas():
        logger.info("Successfully applied all L1 schemas")
    else:
        logger.error("Failed to apply all L1 schemas") 