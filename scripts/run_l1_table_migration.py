#!/usr/bin/env python3
"""
Script to run the L1 tables migration to add version columns.
"""
import os
import sys
import logging
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection parameters - use environment variables
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'second-me-prototype')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')

# Path to migration SQL file
MIGRATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'scripts', 'migrations', 'add_version_to_l1_tables.sql'
)

def run_migration():
    """Run the L1 tables migration to add version columns."""
    start_time = datetime.now()
    logger.info(f"Starting L1 tables migration at {start_time}")
    
    # Check if migration file exists
    if not os.path.exists(MIGRATION_FILE):
        logger.error(f"Migration file not found: {MIGRATION_FILE}")
        return False
    
    try:
        # Read the migration SQL file
        with open(MIGRATION_FILE, 'r') as f:
            migration_sql = f.read()
        
        logger.info(f"Read migration file: {MIGRATION_FILE}")
        
        # Connect to the database
        logger.info(f"Connecting to database {DB_NAME} on {DB_HOST}:{DB_PORT}")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Create a cursor and execute the migration SQL
        with conn.cursor() as cursor:
            logger.info("Executing migration SQL")
            cursor.execute(migration_sql)
            
            # Check for any notices or warnings
            for notice in conn.notices:
                logger.info(f"DB Notice: {notice}")
        
        # Commit the transaction
        conn.commit()
        logger.info("Migration committed successfully")
        
        # Close the connection
        conn.close()
        logger.info("Database connection closed")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Migration completed successfully in {duration:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running migration: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1) 