#!/usr/bin/env python3
"""
Test script to verify the _extract_notes_from_l0 method in L1Manager.
This script tests connecting to real data sources and processing real files.
Updated to use proper dependency injection for all components.
"""
import os
import sys
import logging
import time
from datetime import datetime
import numpy as np
import traceback

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.providers.rel_db import RelationalDB
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter
from app.providers.l1.postgres_adapter import PostgresAdapter
from app.providers.l1.weaviate_adapter import WeaviateAdapter
from app.providers.vector_db import VectorDB
from app.processors.l0.utils import setup_logger
from app.processors.l1.l1_manager import L1Manager
from app.processors.l1.l1_generator import L1Generator
from app.models.l1 import Note, Chunk
from app.processors.l1.topics_generator import TopicsGenerator
from app.processors.l1.shade_generator import ShadeGenerator
from app.processors.l1.shade_merger import ShadeMerger
from app.processors.l1.biography_generator import BiographyGenerator
from app.services.llm_service import LLMService

# Setup logger
logger = setup_logger("run_l1_manager_methods")

def verify_environment():
    """Verify that all environment variables are set correctly"""
    logger.info("Verifying environment variables...")
    
    required_vars = [
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
        'WASABI_ENDPOINT', 'WASABI_ACCESS_KEY', 'WASABI_SECRET_KEY', 
        'WASABI_BUCKET', 'WASABI_REGION',
        'WEAVIATE_URL', 'WEAVIATE_API_KEY', 'EMBEDDING_MODEL'
    ]
    
    for var in required_vars:
        value = getattr(settings, var, None)
        if not value:
            logger.error(f"Missing environment variable: {var}")
        else:
            logger.debug(f"Found environment variable: {var}")
    
    logger.info("Environment verification complete")

def run_extract_notes_from_l0():
    """Test the _extract_notes_from_l0 method of L1Manager with real L0 data"""
    logger.info("Starting test of _extract_notes_from_l0 with REAL adapters...")
    
    # First, verify environment variables
    verify_environment()
    
    # Initialize resources
    rel_db = None
    vector_db = None
    session = None
    
    start_time = time.time()
    
    try:
        #--------------------------------------------------------------------------
        # 1. Create all adapters first (lowest level components)
        #--------------------------------------------------------------------------
        logger.info("STEP 1: Initializing core adapters...")
        
        # 1.1 Initialize PostgreSQL adapter
        logger.info("Connecting to PostgreSQL...")
        rel_db = RelationalDB(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        session = rel_db.get_db_session()
        postgres_adapter = PostgresAdapter(rel_db=rel_db)
        logger.info("Successfully connected to PostgreSQL")
        
        # 1.2 Initialize Wasabi adapter
        logger.info("Connecting to Wasabi S3...")
        logger.debug(f"Using endpoint: {settings.WASABI_ENDPOINT}")
        logger.debug(f"Using bucket: {settings.WASABI_BUCKET}")
        logger.debug(f"Using region: {settings.WASABI_REGION}")
        wasabi_adapter = WasabiStorageAdapter(
            endpoint_url=settings.WASABI_ENDPOINT,
            access_key=settings.WASABI_ACCESS_KEY,
            secret_key=settings.WASABI_SECRET_KEY,
            bucket_name=settings.WASABI_BUCKET,
            region_name=settings.WASABI_REGION
        )
        logger.info("Wasabi adapter initialized")
        
        # 1.3 Initialize Weaviate adapter
        logger.info("Connecting to Weaviate...")
        logger.debug(f"Using URL: {settings.WEAVIATE_URL}")
        vector_db = VectorDB(
            url=settings.WEAVIATE_URL,
            api_key=settings.WEAVIATE_API_KEY,
            embedding_model=settings.EMBEDDING_MODEL
        )
        weaviate_adapter = WeaviateAdapter(client=vector_db.client)
        logger.info("Successfully connected to Weaviate")
        
        # 1.4 Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = LLMService()
        logger.info("LLM service initialized")
        
        #--------------------------------------------------------------------------
        # 2. Create the generator components with proper dependencies
        #--------------------------------------------------------------------------
        logger.info("STEP 2: Creating generator components...")
        
        # 2.1 Create TopicsGenerator
        logger.info("Creating TopicsGenerator...")
        topics_generator = TopicsGenerator(
            llm_service=llm_service
        )
        
        # 2.2 Create ShadeGenerator
        logger.info("Creating ShadeGenerator...")
        shade_generator = ShadeGenerator(
            llm_service=llm_service,
            wasabi_adapter=wasabi_adapter
        )
        
        # 2.3 Create ShadeMerger (depends on ShadeGenerator)
        logger.info("Creating ShadeMerger...")
        shade_merger = ShadeMerger(
            shade_generator=shade_generator
        )
        
        # 2.4 Create BiographyGenerator
        logger.info("Creating BiographyGenerator...")
        biography_generator = BiographyGenerator(
            llm_service=llm_service,
            wasabi_adapter=wasabi_adapter
        )
        
        # 2.5 Create L1Generator (depends on other generators)
        logger.info("Creating L1Generator...")
        l1_generator = L1Generator(
            topics_generator=topics_generator,
            shade_generator=shade_generator,
            biography_generator=biography_generator
        )
        
        #--------------------------------------------------------------------------
        # 3. Initialize L1Manager with all dependencies properly injected
        #--------------------------------------------------------------------------
        logger.info("STEP 3: Initializing L1Manager with all injected dependencies...")
        l1_manager = L1Manager(
            postgres_adapter=postgres_adapter,
            weaviate_adapter=weaviate_adapter,
            wasabi_adapter=wasabi_adapter,
            l1_generator=l1_generator,
            topics_generator=topics_generator,
            shade_generator=shade_generator,
            shade_merger=shade_merger,
            biography_generator=biography_generator
        )
        logger.info("L1Manager initialized successfully")
        
        #--------------------------------------------------------------------------
        # 4. Run the test method
        #--------------------------------------------------------------------------
        user_id = settings.DEFAULT_USER_ID
        logger.info(f"STEP 4: Calling _extract_notes_from_l0 for user {user_id}...")
        
        notes_list, memory_list = l1_manager._extract_notes_from_l0(user_id)
        
        #--------------------------------------------------------------------------
        # 5. Validate and display the results
        #--------------------------------------------------------------------------
        logger.info(f"STEP 5: Processing results...")
        logger.info(f"Extracted {len(notes_list)} notes and {len(memory_list)} memory items")
        
        if not notes_list or not memory_list:
            logger.warning("No notes or memory items found. This could be an error or there might be no data.")
            logger.info("\n⚠️ _extract_notes_from_l0 test completed, but no data was found")
            return True
        
        # Display results for first few notes
        for i, note in enumerate(notes_list[:3]):  # Show first 3 notes
            logger.info(f"\nNote {i+1}:")
            logger.info(f"  ID: {note.id}")
            logger.info(f"  Title: {note.title}")
            logger.info(f"  Create time: {note.create_time}")
            logger.info(f"  Number of chunks: {len(note.chunks)}")
            logger.info(f"  Memory type: {note.memory_type}")
            logger.info(f"  Content length: {len(note.content)}")
            
            # Test compatibility properties
            logger.info(f"  Compatibility properties:")
            logger.info(f"    noteId matches id: {note.id == note.noteId}")
            logger.info(f"    createTime matches create_time: {note.create_time == note.createTime}")
            logger.info(f"    memoryType matches memory_type: {note.memory_type == note.memoryType}")
            
            # Log first couple of chunks
            for j, chunk in enumerate(note.chunks[:2]):  # Show first 2 chunks
                logger.info(f"  Chunk {j+1}:")
                logger.info(f"    ID: {chunk.id}")
                logger.info(f"    Content length: {len(chunk.content)}")
                
                # Test squeeze method works if embedding exists
                if chunk.embedding is not None:
                    embedding_shape = np.array(chunk.embedding).shape
                    logger.info(f"    Embedding shape: {embedding_shape}")
                    try:
                        squeezed = chunk.squeeze()
                        logger.info(f"    Squeeze method works: {squeezed is not None}")
                    except Exception as e:
                        logger.error(f"    Error calling squeeze: {str(e)}")
                else:
                    logger.warning(f"    No embedding for chunk {chunk.id}")
        
        # Display results for first few memory items
        for i, memory in enumerate(memory_list[:3]):  # Show first 3 memory items
            logger.info(f"\nMemory {i+1}:")
            logger.info(f"  memoryId: {memory.get('memoryId')}")
            logger.info(f"  embedding shape: {len(memory.get('embedding', [])) if 'embedding' in memory else 'None'}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ _extract_notes_from_l0 test completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Error testing _extract_notes_from_l0 after {elapsed_time:.2f} seconds: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Clean up resources
        logger.info("\nCleaning up resources...")
        if session:
            try:
                rel_db.close_db_session(session)
                logger.info("  Closed database session")
            except Exception as e:
                logger.error(f"  Error closing database session: {str(e)}")
                
        if vector_db:
            try:
                vector_db.close()
                logger.info("  Closed vector database connection")
            except Exception as e:
                logger.error(f"  Error closing vector database: {str(e)}")

def test_generate_l1_from_l0():
    """Test the full generation process with the generate_l1_from_l0 method"""
    logger.info("Starting test of generate_l1_from_l0...")
    # This could be implemented later to test the full generation process
    logger.info("Not implemented yet")
    return True

if __name__ == "__main__":
    # Set up logger first
    logger.info("=== Starting L1Manager methods test with proper dependency injection ===")
    
    # Run the test of _extract_notes_from_l0
    notes_test_success = run_extract_notes_from_l0()
    
    # Final results
    if notes_test_success:
        logger.info("All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1) 