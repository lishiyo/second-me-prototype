#!/usr/bin/env python3
"""
Script to fix the TenantCluster schema by adding the missing s3_path property.
This script should be run to fix existing deployments where the TenantCluster
schema was created without the s3_path property.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the app directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.providers.vector_db import VectorDB
from app.models.l1.schema_application import add_s3_path_to_tenant_cluster

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the schema fix.
    """
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize VectorDB
        logger.info("Initializing VectorDB connection")
        vector_db = VectorDB(
            url=os.environ.get("WEAVIATE_URL"),
            api_key=os.environ.get("WEAVIATE_API_KEY"),
            embedding_model=os.environ.get("EMBEDDING_MODEL")
        )
        
        # Add s3_path property to TenantCluster collection
        logger.info("Running TenantCluster schema fix")
        success = add_s3_path_to_tenant_cluster(vector_db)
        
        if success:
            logger.info("✅ Schema fix completed successfully")
            return 0
        else:
            logger.error("❌ Schema fix failed")
            return 1
    
    except Exception as e:
        logger.error(f"Error running schema fix: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 