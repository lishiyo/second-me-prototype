#!/usr/bin/env python3
"""
Apply the L1Shade migration to the PostgreSQL database.

This script applies the ALTER_SHADES_TABLE migration to add the new L1Shade 
fields to the existing shades table and updates existing records.
"""
import os
import sys
import logging
from sqlalchemy import text

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.providers.rel_db import RelationalDB
from app.models.l1.migrations import ALTER_SHADES_TABLE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_migration():
    """Apply the L1Shade migration."""
    logger.info("Connecting to PostgreSQL...")
    rel_db = RelationalDB(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )
    
    session = rel_db.get_db_session()
    
    try:
        logger.info("Applying ALTER_SHADES_TABLE migration...")
        # Use text() to wrap the SQL statement
        session.execute(text(ALTER_SHADES_TABLE))
        session.commit()
        logger.info("Migration applied successfully!")
        
        # Check if the migration worked
        logger.info("Verifying migration...")
        results = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'shades';")).fetchall()
        columns = [r[0] for r in results]
        
        expected_columns = ['aspect', 'icon', 'desc_second_view', 'desc_third_view', 
                           'content_second_view', 'content_third_view']
        
        missing_columns = [col for col in expected_columns if col not in columns]
        
        if missing_columns:
            logger.error(f"Migration incomplete! Missing columns: {missing_columns}")
        else:
            logger.info("Migration verification successful. All columns added.")
            
        # Check if content_third_view was updated from summary
        count = session.execute(
            text("SELECT COUNT(*) FROM shades WHERE content_third_view IS NOT NULL;")
        ).scalar()
        
        logger.info(f"Found {count} shades with content_third_view populated.")
            
    except Exception as e:
        logger.error(f"Error applying migration: {str(e)}")
        session.rollback()
        raise
    finally:
        rel_db.close_db_session(session)
        
if __name__ == "__main__":
    logger.info("Starting L1Shade migration application...")
    apply_migration()
    logger.info("L1Shade migration application complete!") 