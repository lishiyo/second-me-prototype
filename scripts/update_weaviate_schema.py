#!/usr/bin/env python
"""
Script to update Weaviate schema with version property for L1 clusters.
Run this after updating the schema definition in weaviate_adapter.py.

Usage:
    python scripts/update_weaviate_schema.py [--env-file PATH_TO_ENV_FILE]

Example:
    python scripts/update_weaviate_schema.py --env-file .env.local
"""
import os
import sys
import logging
import argparse
import json
import requests
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports from app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from app.providers.l1.weaviate_adapter import WeaviateAdapter
from app.providers.vector_db import VectorDB

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update Weaviate schema with version property for L1 clusters')
    parser.add_argument('--env-file', help='Path to .env file (default: .env)', default='.env')
    return parser.parse_args()

def check_version_property_via_http(url, api_key, collection_name="TenantCluster"):
    """
    Use direct HTTP API to check if version property exists in the collection.
    
    Args:
        url: Weaviate URL
        api_key: Weaviate API key
        collection_name: Collection to check
        
    Returns:
        True if version property exists, False otherwise
    """
    try:
        # Make sure URL has proper format
        if not url.startswith("http"):
            url = f"https://{url}"
        
        # Add API endpoint path for v4
        schema_url = f"{url}/v1/schema/{collection_name}"
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Make the request
        response = requests.get(schema_url, headers=headers)
        
        if response.status_code == 200:
            # Parse the response
            data = response.json()
            logger.info(f"HTTP API response: {data}")
            
            # Look for version property
            if "properties" in data:
                for prop in data["properties"]:
                    if prop.get("name") == "version":
                        logger.info("Found version property via HTTP API check")
                        return True
            
            logger.info("Version property not found via HTTP API check")
            return False
        else:
            logger.error(f"HTTP API request failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error checking version property via HTTP: {e}")
        return False

def add_version_property_via_http(url, api_key, collection_name="TenantCluster"):
    """
    Use direct HTTP API to add version property to the collection.
    
    Args:
        url: Weaviate URL
        api_key: Weaviate API key
        collection_name: Collection to update
        
    Returns:
        True if version property was added, False otherwise
    """
    try:
        # Make sure URL has proper format
        if not url.startswith("http"):
            url = f"https://{url}"
        
        # Add API endpoint path for v4
        property_url = f"{url}/v1/schema/{collection_name}/properties"
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Define the property for v4
        property_data = {
            "name": "version",
            "dataType": ["number"],
            "description": "L1 version number for filtering",
            "moduleConfig": {
                "text2vec-openai": {
                    "skip": True
                }
            },
            "tokenization": "none",
            "indexFilterable": True,
            "indexSearchable": False
        }
        
        # Try with alternative property definition if the first one fails
        alternative_property_data = {
            "dataType": ["number"],
            "description": "L1 version number for filtering",
            "name": "version",
            "indexFilterable": True,
            "indexSearchable": False
        }
        
        # Make the request
        logger.info(f"Sending HTTP request to add property 'version' to {collection_name}")
        response = requests.post(property_url, headers=headers, json=property_data)
        
        if response.status_code in (200, 201):
            logger.info("Successfully added version property via HTTP API")
            return True
        else:
            logger.warning(f"First attempt failed: {response.status_code}: {response.text}")
            logger.info("Trying alternative property definition")
            response = requests.post(property_url, headers=headers, json=alternative_property_data)
            
            if response.status_code in (200, 201):
                logger.info("Successfully added version property via HTTP API using alternative definition")
                return True
            else:
                logger.error(f"HTTP API request failed with status {response.status_code}: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Error adding version property via HTTP: {e}")
        return False

def update_schema():
    """Update the Weaviate schema with new version property."""
    logger.info("Starting Weaviate schema update...")
    
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
        
        # Create WeaviateAdapter with the configured client
        adapter = WeaviateAdapter(client=client)
        
        # Apply the updated schema
        success = adapter.apply_schema()
        
        if success:
            logger.info("Successfully updated Weaviate schema with version property")
        else:
            logger.error("Failed to update Weaviate schema")
            return 1
        
        # Verify that the schema was updated
        try:
            # Get list of all collections (in v4 we check collections instead of classes)
            collections = client.collections.list_all()
            logger.info(f"Found collections: {collections}")
            
            # Check if TenantCluster collection exists
            if "TenantCluster" in collections:
                # Get the collection details
                cluster_collection = client.collections.get("TenantCluster")
                
                try:
                    # In Weaviate v4, we need to extract properties from the _CollectionConfigSimple directly
                    # Examining the logs, we can see collections are returned as _CollectionConfigSimple objects
                    # with a 'properties' list containing _Property objects
                    
                    # First, try to access it directly from the collections dict
                    logger.info("Checking for version property in TenantCluster properties")
                    has_version_property = False
                    
                    # Get the TenantCluster config from the collections dictionary
                    collection_config = collections.get("TenantCluster")
                    if collection_config and hasattr(collection_config, "properties"):
                        # Examine each property
                        for prop in collection_config.properties:
                            logger.info(f"Found property: {prop.name} of type {prop.data_type}")
                            if prop.name == "version":
                                has_version_property = True
                                logger.info("Verified: version property was added to TenantCluster collection")
                                break
                    
                    if not has_version_property:
                        # If version not found, try to add it using properties API
                        logger.warning("Version property not found in TenantCluster collection. Attempting to add it now.")
                        try:
                            from weaviate.classes.config import Property, DataType
                            # Try different ways to add a property in Weaviate v4
                            
                            # Method 1: Using properties API on collection
                            if hasattr(cluster_collection, "properties"):
                                cluster_collection.properties.create(
                                    Property(
                                        name="version",
                                        description="L1 version number for filtering",
                                        data_type=DataType.NUMBER,
                                        indexing={"filterable": True, "searchable": False},
                                    )
                                )
                                logger.info("Successfully added version property to TenantCluster collection")
                            # Method 2: Using schema API directly
                            else:
                                logger.info("Using schema API to add property")
                                # In Weaviate v4, we use the collection API instead of the schema API
                                try:
                                    # Get collection and create property
                                    from weaviate.classes.config import Property, DataType
                                    collection = client.collections.get("TenantCluster")
                                    
                                    # Try adding property via API
                                    if hasattr(collection, "properties") and hasattr(collection.properties, "create"):
                                        collection.properties.create(
                                            Property(
                                                name="version",
                                                description="L1 version number for filtering",
                                                data_type=DataType.NUMBER,
                                                indexing={"filterable": True, "searchable": False}
                                            )
                                        )
                                        logger.info("Successfully added version property to collection")
                                    else:
                                        # Fall back to direct HTTP API
                                        logger.info("Using direct HTTP API to add property")
                                        add_version_property_via_http(weaviate_url, weaviate_api_key)
                                except Exception as e:
                                    logger.error(f"Error adding property via collection API: {e}")
                                    # Fall back to direct HTTP API
                                    logger.info("Falling back to direct HTTP API")
                                    add_version_property_via_http(weaviate_url, weaviate_api_key)
                        except Exception as e:
                            logger.error(f"Error adding version property: {e}")
                except Exception as e:
                    logger.error(f"Error accessing properties: {e}")
                    logger.info("Attempting alternative approach to add version property")
                    # Try HTTP API as a direct fallback
                    try:
                        logger.info("Using direct HTTP API")
                        add_version_property_via_http(weaviate_url, weaviate_api_key)
                    except Exception as e2:
                        logger.error(f"Error in HTTP API approach: {e2}")
                        logger.warning("Could not add version property to TenantCluster collection")
            else:
                logger.warning("TenantCluster collection not found")
            
            # Properly close the connection
            client.close()
            logger.info("Closed Weaviate client connection")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error verifying schema update: {e}")
            # Try to close the connection even if verification failed
            client.close()
            return 1
    
    except Exception as e:
        logger.error(f"Error initializing Weaviate client: {e}")
        return 1
    
    # Manual instructions if all else fails
    logger.info("\n" + "="*80)
    logger.info("If the automatic property addition failed, you can manually add the 'version' property:")
    logger.info("1. Visit the Weaviate Cloud Console at https://console.weaviate.cloud/")
    logger.info("2. Select your cluster")
    logger.info("3. Go to the Schema tab")
    logger.info("4. Find the 'TenantCluster' collection")
    logger.info("5. Click 'Add property'")
    logger.info("6. Create a property with:")
    logger.info("   - Name: version")
    logger.info("   - Data type: number")
    logger.info("   - Description: L1 version number for filtering")
    logger.info("   - Check 'Make filterable'")
    logger.info("   - Uncheck 'Make searchable'")
    logger.info("="*80)
    
    # The script succeeded even if we couldn't add the property
    # The user can follow the manual instructions if needed
    return 0

if __name__ == "__main__":
    sys.exit(update_schema()) 