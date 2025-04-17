#!/usr/bin/env python
"""
Script to migrate Weaviate collections to support multi-tenancy.
This will:
1. Back up existing data from TenantCluster collection
2. Delete the existing collection
3. Recreate the collection with multi-tenancy enabled
4. Restore the data with tenant information

Usage:
    python scripts/migrate_to_weaviate_multitenancy.py [--env-file PATH_TO_ENV_FILE]

Example:
    python scripts/migrate_to_weaviate_multitenancy.py --env-file .env.local
"""
import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports from app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from app.providers.l1.weaviate_adapter import WeaviateAdapter, CLUSTERS_COLLECTION
from app.providers.vector_db import VectorDB

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Migrate Weaviate collections to support multi-tenancy')
    parser.add_argument('--env-file', help='Path to .env file (default: .env)', default='.env')
    parser.add_argument('--backup-dir', help='Directory to store backups (default: ./weaviate_backup)', default='./weaviate_backup')
    parser.add_argument('--skip-backup', help='Skip backing up data (use with caution)', action='store_true')
    parser.add_argument('--force', help='Force migration even if there are warnings', action='store_true')
    return parser.parse_args()

def backup_collection_data(client, collection_name: str, backup_dir: str) -> str:
    """
    Back up data from a Weaviate collection.
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection to back up
        backup_dir: Directory to store backup
        
    Returns:
        Path to the backup file
    """
    try:
        logger.info(f"Backing up data from collection {collection_name}...")
        
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"{collection_name}_{timestamp}.json")
        
        # Get collection
        collection = client.collections.get(collection_name)
        
        # Get all objects
        try:
            # Use newer API
            result = collection.query.fetch_objects(limit=10000)
            objects = result.objects
            
            # Convert objects to a serializable format
            backup_data = []
            for obj in objects:
                # Extract vector if available
                vector = obj.vector if hasattr(obj, "vector") else None
                
                # Create serializable object
                backup_obj = {
                    "uuid": obj.uuid,
                    "properties": obj.properties,
                    "vector": vector
                }
                
                # Try to extract user_id from properties, this will be used as tenant
                if "user_id" in obj.properties:
                    backup_obj["tenant"] = obj.properties["user_id"]
                
                backup_data.append(backup_obj)
            
        except (AttributeError, ImportError) as e:
            # Fall back to legacy API
            logger.warning(f"Error using v4 API for backup, falling back to legacy mode: {e}")
            result = client.query.get(collection_name).with_additional(["id", "vector"]).with_limit(10000).do()
            objects = result.get("data", {}).get("Get", {}).get(collection_name, [])
            
            # Convert objects to a serializable format
            backup_data = []
            for obj in objects:
                # Extract vector if available
                vector = obj.get("_additional", {}).get("vector")
                
                # Create serializable object
                backup_obj = {
                    "uuid": obj.get("_additional", {}).get("id"),
                    "properties": {k: v for k, v in obj.items() if not k.startswith("_")},
                    "vector": vector
                }
                
                # Try to extract user_id from properties, this will be used as tenant
                if "user_id" in backup_obj["properties"]:
                    backup_obj["tenant"] = backup_obj["properties"]["user_id"]
                
                backup_data.append(backup_obj)
        
        # Write backup to file
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Backed up {len(backup_data)} objects to {backup_file}")
        return backup_file
    
    except Exception as e:
        logger.error(f"Error backing up collection {collection_name}: {e}")
        raise

def delete_collection(client, collection_name: str) -> bool:
    """
    Delete a Weaviate collection.
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection to delete
        
    Returns:
        True if collection was deleted, False otherwise
    """
    try:
        logger.info(f"Deleting collection {collection_name}...")
        
        # Check if collection exists
        collections = client.collections.list_all()
        if collection_name not in collections:
            logger.warning(f"Collection {collection_name} does not exist, skipping deletion")
            return True
        
        # Delete collection
        client.collections.delete(collection_name)
        logger.info(f"Deleted collection {collection_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        return False

def create_collection_with_multitenancy(client, collection_name: str) -> bool:
    """
    Create a Weaviate collection with multi-tenancy enabled.
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection to create
        
    Returns:
        True if collection was created, False otherwise
    """
    try:
        logger.info(f"Creating collection {collection_name} with multi-tenancy...")
        
        # Import necessary modules
        from weaviate.classes.config import Configure, Property, DataType
        
        # Create collection
        client.collections.create(
            name=collection_name,
            description="L1 Clusters with multi-tenancy",
            # Use pre-computed embeddings from OpenAI rather than Weaviate's vectorizer
            vectorizer_config=Configure.Vectorizer.none(),
            # Enable multi-tenancy
            multi_tenancy_config=Configure.multi_tenancy(
                enabled=True,
                auto_tenant_creation=True
            ),
            properties=[
                Property(
                    name="user_id",
                    description="User ID",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="cluster_id",
                    description="Cluster ID",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="topic_id",
                    description="Parent topic ID",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="name",
                    description="Cluster name",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="summary",
                    description="Cluster summary",
                    data_type=DataType.TEXT,
                    indexing={"filterable": False, "searchable": True},
                ),
                Property(
                    name="metadata",
                    description="JSON-encoded metadata",
                    data_type=DataType.TEXT,
                    indexing={"filterable": False, "searchable": False},
                ),
                Property(
                    name="s3_path",
                    description="Path to full content in Wasabi S3",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": False},
                ),
                Property(
                    name="version",
                    description="L1 version number for filtering",
                    data_type=DataType.NUMBER,
                    indexing={"filterable": True, "searchable": False},
                ),
            ]
        )
        
        logger.info(f"Created collection {collection_name} with multi-tenancy")
        return True
    
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        return False

def restore_collection_data(client, collection_name: str, backup_file: str) -> bool:
    """
    Restore data to a Weaviate collection with tenants.
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection to restore
        backup_file: Path to the backup file
        
    Returns:
        True if data was restored, False otherwise
    """
    try:
        logger.info(f"Restoring data to collection {collection_name} from {backup_file}...")
        
        # Read backup file
        with open(backup_file, "r") as f:
            backup_data = json.load(f)
        
        # Get collection
        collection = client.collections.get(collection_name)
        
        # Group objects by tenant
        tenant_objects = {}
        for obj in backup_data:
            tenant = obj.get("tenant")
            if not tenant:
                logger.warning(f"Object {obj.get('uuid')} has no tenant, skipping: {obj}")
                continue
            
            if tenant not in tenant_objects:
                tenant_objects[tenant] = []
            
            tenant_objects[tenant].append(obj)
        
        # Insert objects by tenant
        for tenant, objects in tenant_objects.items():
            logger.info(f"Restoring {len(objects)} objects for tenant {tenant}...")
            
            # Get tenant-specific collection
            tenant_collection = collection.with_tenant(tenant)
            
            # Insert objects
            for obj in objects:
                try:
                    uuid = obj.get("uuid")
                    properties = obj.get("properties", {})
                    vector = obj.get("vector")
                    
                    # Insert object
                    tenant_collection.data.insert(
                        uuid=uuid,
                        properties=properties,
                        vector=vector
                    )
                except Exception as e:
                    logger.error(f"Error inserting object {obj.get('uuid')}: {e}")
        
        logger.info(f"Restored data to collection {collection_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error restoring collection {collection_name}: {e}")
        return False

def migrate_collection_to_multitenancy(client, collection_name: str, backup_dir: str, skip_backup: bool = False) -> bool:
    """
    Migrate a Weaviate collection to support multi-tenancy.
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection to migrate
        backup_dir: Directory to store backups
        skip_backup: Whether to skip backing up data
        
    Returns:
        True if migration was successful, False otherwise
    """
    backup_file = None
    
    try:
        # Step 1: Back up existing data
        if not skip_backup:
            backup_file = backup_collection_data(client, collection_name, backup_dir)
        else:
            logger.warning("Skipping backup as requested")
        
        # Step 2: Delete existing collection
        if not delete_collection(client, collection_name):
            logger.error(f"Failed to delete collection {collection_name}, aborting migration")
            return False
        
        # Step 3: Recreate collection with multi-tenancy
        if not create_collection_with_multitenancy(client, collection_name):
            logger.error(f"Failed to create collection {collection_name} with multi-tenancy, aborting migration")
            return False
        
        # Step 4: Restore data with tenant information
        if backup_file and not skip_backup:
            if not restore_collection_data(client, collection_name, backup_file):
                logger.error(f"Failed to restore data to collection {collection_name}, migration completed but data was not restored")
                return False
        
        logger.info(f"Successfully migrated collection {collection_name} to support multi-tenancy")
        return True
    
    except Exception as e:
        logger.error(f"Error migrating collection {collection_name}: {e}")
        return False

def migrate_weaviate_to_multitenancy():
    """Migrate Weaviate collections to support multi-tenancy."""
    logger.info("Starting Weaviate multi-tenancy migration...")
    
    args = parse_arguments()
    
    # Load environment variables from .env file
    env_path = Path(args.env_file)
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logger.warning(f"Environment file {env_path} not found, using existing environment variables")
    
    # Check for environment variables
    weaviate_url = os.environ.get('WEAVIATE_URL')
    weaviate_api_key = os.environ.get('WEAVIATE_API_KEY')
    
    if not weaviate_url:
        logger.error("WEAVIATE_URL environment variable is not set in the .env file")
        return 1
    
    if not weaviate_api_key:
        logger.error("WEAVIATE_API_KEY environment variable is not set in the .env file")
        return 1
    
    logger.info(f"Using Weaviate URL: {weaviate_url}")
    logger.info("Weaviate API key found in environment variables")
    
    # IMPORTANT WARNING
    logger.warning("="*80)
    logger.warning("IMPORTANT: This script will migrate your Weaviate collections to support multi-tenancy.")
    logger.warning("This will DELETE your existing collections and recreate them with multi-tenancy enabled.")
    logger.warning("Data will be backed up and restored, but this is a destructive operation.")
    logger.warning("Make sure you have a recent backup of your data before proceeding.")
    logger.warning("="*80)
    
    if not args.force:
        confirm = input("Do you want to proceed with the migration? [y/N] ")
        if confirm.lower() != "y":
            logger.info("Migration aborted by user")
            return 0
    
    try:
        # Create a Weaviate v4 client directly
        import weaviate
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key=weaviate_api_key),
            headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY', '')
            }
        )
        
        # Migrate TenantCluster collection
        if not migrate_collection_to_multitenancy(client, CLUSTERS_COLLECTION, args.backup_dir, args.skip_backup):
            logger.error(f"Failed to migrate collection {CLUSTERS_COLLECTION}")
            return 1
        
        # Properly close the connection
        client.close()
        logger.info("Closed Weaviate client connection")
        
        logger.info("Weaviate multi-tenancy migration completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error during Weaviate multi-tenancy migration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(migrate_weaviate_to_multitenancy()) 