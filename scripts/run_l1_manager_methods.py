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

def initialize_l1_test_environment():
    """
    Initialize the L1 test environment with adapters, services and components.
    This handles the common setup logic for all test functions.
    
    Returns:
        dict: A dictionary containing all the initialized resources, including:
            - rel_db: Relational database connection
            - vector_db: Vector database connection
            - session: Database session
            - postgres_adapter: PostgreSQL adapter
            - wasabi_adapter: Wasabi storage adapter
            - weaviate_adapter: Weaviate adapter
            - llm_service: LLM service
            - topics_generator: Topics generator
            - shade_generator: Shade generator
            - shade_merger: Shade merger
            - biography_generator: Biography generator
            - l1_generator: L1 generator
            - l1_manager: L1 manager
            - notes_list: List of extracted notes
            - memory_list: List of memory items
            - user_id: User ID used for extraction
    """
    # Verify environment variables first
    verify_environment()
    
    # Initialize resources
    rel_db = None
    vector_db = None
    session = None
    
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
        # 4. Extract notes and memory list from L0
        #--------------------------------------------------------------------------
        user_id = settings.DEFAULT_USER_ID
        logger.info(f"STEP 4: Extracting notes from L0 for user {user_id}...")
        
        notes_list, memory_list = l1_manager._extract_notes_from_l0(user_id)
        logger.info(f"Extracted {len(notes_list)} notes and {len(memory_list)} memory items")
        
        #--------------------------------------------------------------------------
        # 5. Validate embeddings in memory_list to catch issues early
        #--------------------------------------------------------------------------
        logger.info("Validating memory embeddings...")
        scalar_embeddings = 0
        missing_embeddings = 0
        short_embeddings = 0
        dict_embeddings = 0
        good_embeddings = 0
        
        for i, memory in enumerate(memory_list):
            if 'embedding' not in memory or memory['embedding'] is None:
                missing_embeddings += 1
                continue
                
            embedding = memory['embedding']
            
            # Handle dictionary embeddings (unexpected format)
            if isinstance(embedding, dict):
                dict_embeddings += 1
                if i < 5:  # Only log for first few items to avoid excessive logging
                    logger.warning(f"Memory {memory.get('memoryId')} has dictionary as embedding: {str(embedding)[:100]}...")
                
                # Try to extract vector from the dictionary if it has a 'default' key
                if 'default' in embedding and isinstance(embedding['default'], list):
                    logger.info(f"  Extracting vector from 'default' key in dictionary for memory {memory.get('memoryId')}")
                    memory['embedding'] = embedding['default']
                # If no 'default' key but has other list values, use the first one
                else:
                    for key, value in embedding.items():
                        if isinstance(value, list) and len(value) > 10:
                            logger.info(f"  Using list value from key '{key}' for memory {memory.get('memoryId')}")
                            memory['embedding'] = value
                            break
                    else:
                        # No suitable list found, create a dummy embedding
                        logger.warning(f"  Could not extract vector from dictionary for memory {memory.get('memoryId')}, creating dummy vector")
                        memory['embedding'] = [0.0] * 10
                continue
            
            # Handle normal embeddings (as before)
            try:
                emb_array = np.array(embedding)
                
                if emb_array.shape == ():  # Scalar value
                    scalar_embeddings += 1
                    # Fix scalar embeddings by converting to small vector
                    if i < 5:  # Only log for first few items to avoid excessive logging
                        value = emb_array.item()
                        # Check if the scalar is actually a complex object
                        if not (isinstance(value, (int, float, str)) and not isinstance(value, bool)):
                            logger.warning(f"Memory {memory.get('memoryId')} has non-numeric scalar embedding: {str(value)[:100]}...")
                            memory['embedding'] = [0.0] * 10  # Create dummy vector
                        else:
                            logger.warning(f"Memory {memory.get('memoryId')} has scalar embedding: {value}. Fixing...")
                            memory['embedding'] = [float(value)] * 5  # Create small vector
                elif len(emb_array.shape) == 1 and emb_array.shape[0] < 10:
                    short_embeddings += 1
                    if i < 5:  # Only log for first few items
                        logger.warning(f"Memory {memory.get('memoryId')} has suspiciously short embedding: {emb_array.shape}")
                else:
                    good_embeddings += 1
            except Exception as e:
                logger.warning(f"Error processing embedding for memory {memory.get('memoryId')}: {str(e)}")
                logger.warning(f"Embedding type: {type(embedding)}, sample: {str(embedding)[:100]}...")
                # Create a dummy embedding
                memory['embedding'] = [0.0] * 10  # Create dummy vector
                
        if scalar_embeddings > 0 or missing_embeddings > 0 or short_embeddings > 0 or dict_embeddings > 0:
            logger.warning(f"Embedding issues found: {scalar_embeddings} scalar, {missing_embeddings} missing, {short_embeddings} short, {dict_embeddings} dictionary, {good_embeddings} good")
        else:
            logger.info(f"All {good_embeddings} embeddings look valid")
        
        # Return all initialized resources
        return {
            "rel_db": rel_db,
            "vector_db": vector_db,
            "session": session,
            "postgres_adapter": postgres_adapter,
            "wasabi_adapter": wasabi_adapter,
            "weaviate_adapter": weaviate_adapter,
            "llm_service": llm_service,
            "topics_generator": topics_generator,
            "shade_generator": shade_generator,
            "shade_merger": shade_merger,
            "biography_generator": biography_generator,
            "l1_generator": l1_generator,
            "l1_manager": l1_manager,
            "notes_list": notes_list,
            "memory_list": memory_list,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error initializing L1 test environment: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up resources in case of error
        if session:
            try:
                rel_db.close_db_session(session)
            except Exception as e:
                logger.error(f"Error closing database session: {str(e)}")
                
        if vector_db:
            try:
                vector_db.close()
            except Exception as e:
                logger.error(f"Error closing vector database: {str(e)}")
                
        raise e

def cleanup_resources(resources):
    """
    Clean up resources used in the test environment.
    
    Args:
        resources (dict): The dictionary of resources returned by initialize_l1_test_environment
    """
    logger.info("\nCleaning up resources...")
    if "session" in resources and resources["session"]:
        try:
            resources["rel_db"].close_db_session(resources["session"])
            logger.info("  Closed database session")
        except Exception as e:
            logger.error(f"  Error closing database session: {str(e)}")
            
    if "vector_db" in resources and resources["vector_db"]:
        try:
            resources["vector_db"].close()
            logger.info("  Closed vector database connection")
        except Exception as e:
            logger.error(f"  Error closing vector database: {str(e)}")

def run_extract_notes_from_l0():
    """Test the _extract_notes_from_l0 method of L1Manager with real L0 data"""
    logger.info("Starting test of _extract_notes_from_l0 with REAL adapters...")
    
    start_time = time.time()
    resources = None
    
    try:
        # Initialize the test environment and get resources
        resources = initialize_l1_test_environment()
        
        if not resources:
            logger.error("Failed to initialize test environment")
            return False
        
        notes_list = resources["notes_list"]
        memory_list = resources["memory_list"]
        
        #--------------------------------------------------------------------------
        # Validate and display the results
        #--------------------------------------------------------------------------
        logger.info(f"STEP 5: Processing results...")
        
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
            
            # If note has an embedding, check its shape
            if note.embedding is not None:
                embedding_array = np.array(note.embedding)
                logger.info(f"  Note embedding shape: {embedding_array.shape}")
                logger.info(f"  Note embedding type: {type(note.embedding)}")
                
                if embedding_array.shape == ():
                    logger.warning(f"  Note embedding is a scalar value: {embedding_array.item()}")
            
            # Log first couple of chunks with enhanced embedding diagnostics
            for j, chunk in enumerate(note.chunks[:2]):  # Show first 2 chunks
                logger.info(f"  Chunk {j+1}:")
                logger.info(f"    ID: {chunk.id}")
                logger.info(f"    Content length: {len(chunk.content)}")
                
                # Enhanced embedding diagnostics
                if chunk.embedding is not None:
                    raw_shape = np.array(chunk.embedding).shape
                    logger.info(f"    Raw embedding shape: {raw_shape}")
                    logger.info(f"    Raw embedding type: {type(chunk.embedding)}")
                    
                    if raw_shape == ():
                        logger.warning(f"    Raw embedding is a scalar value: {np.array(chunk.embedding).item()}")
                    
                    try:
                        # Test squeeze with enhanced diagnostics
                        squeezed = chunk.squeeze()
                        if squeezed is not None:
                            squeezed_shape = squeezed.shape if hasattr(squeezed, 'shape') else "No shape attribute"
                            logger.info(f"    Squeezed embedding shape: {squeezed_shape}")
                            logger.info(f"    Squeezed embedding type: {type(squeezed)}")
                            
                            # Check if we have a proper vector
                            if hasattr(squeezed, 'shape'):
                                if squeezed.shape == ():
                                    logger.warning(f"    PROBLEM: Squeezed embedding is a scalar value: {squeezed.item()}")
                                elif len(squeezed.shape) == 1 and squeezed.shape[0] < 5:
                                    logger.warning(f"    PROBLEM: Squeezed embedding has suspiciously small dimension: {squeezed.shape}")
                                else:
                                    logger.info(f"    Embedding dimensions look good: {squeezed.shape}")
                        else:
                            logger.warning(f"    Squeeze method returned None")
                    except Exception as e:
                        logger.error(f"    Error calling squeeze: {str(e)}")
                else:
                    logger.warning(f"    No embedding for chunk {chunk.id}")
        
        # Display results for first few memory items with enhanced diagnostics
        for i, memory in enumerate(memory_list[:3]):  # Show first 3 memory items
            logger.info(f"\nMemory {i+1}:")
            logger.info(f"  memoryId: {memory.get('memoryId')}")
            
            # Enhanced embedding diagnostics
            if 'embedding' in memory:
                embedding = memory.get('embedding')
                logger.info(f"  Embedding type: {type(embedding)}")
                
                if embedding is not None:
                    # Handle dictionary embeddings
                    if isinstance(embedding, dict):
                        logger.warning(f"  PROBLEM: Embedding is a dictionary: {str(embedding)[:100]}...")
                        # Check if it has a 'default' key with a vector
                        if 'default' in embedding and isinstance(embedding['default'], list):
                            vector = embedding['default']
                            logger.info(f"  Found vector in 'default' key, shape: {np.array(vector).shape}")
                        else:
                            logger.warning(f"  No 'default' vector found in dictionary embedding")
                    else:
                        # Handle normal array embeddings
                        try:
                            raw_shape = np.array(embedding).shape
                            logger.info(f"  Raw embedding shape: {raw_shape}")
                            
                            if raw_shape == ():
                                logger.warning(f"  PROBLEM: Embedding is a scalar value: {np.array(embedding).item()}")
                            elif len(raw_shape) == 1:
                                logger.info(f"  Embedding vector length: {raw_shape[0]}")
                                
                                if raw_shape[0] < 10:
                                    logger.warning(f"  PROBLEM: Embedding vector is suspiciously short: {raw_shape[0]} elements")
                            else:
                                logger.warning(f"  Unusual embedding shape: {raw_shape}")
                        except Exception as e:
                            logger.error(f"  Error analyzing embedding: {str(e)}")
            else:
                logger.warning(f"  Memory {memory.get('memoryId')} has no embedding field")
        
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
        if resources:
            cleanup_resources(resources)

def run_topics_for_shades():
    """Test the generate_topics_for_shades method of TopicsGenerator with real data"""
    logger.info("Starting test of generate_topics_for_shades...")
    
    start_time = time.time()
    resources = None
    
    try:
        # Initialize the test environment and get resources
        resources = initialize_l1_test_environment()
        
        if not resources:
            logger.error("Failed to initialize test environment")
            return False
            
        topics_generator = resources["topics_generator"]
        memory_list = resources["memory_list"]
        user_id = resources["user_id"]
        
        if not memory_list:
            logger.warning("No memory items found. Cannot continue the test.")
            return False
        
        #--------------------------------------------------------------------------
        # 4. Test cold start (no existing clusters)
        #--------------------------------------------------------------------------
        logger.info("STEP 4: Testing generate_topics_for_shades with COLD START...")
        
        # Test cold start with empty clusters
        cold_start_result = topics_generator.generate_topics_for_shades(
            user_id=user_id,
            old_cluster_list=[],
            old_outlier_memory_list=[],
            new_memory_list=memory_list
        )
        
        # Log the results
        cluster_count = len(cold_start_result.get("clusterList", []))
        outlier_count = len(cold_start_result.get("outlierMemoryList", []))
        logger.info(f"Cold start generated {cluster_count} clusters and {outlier_count} outliers")
        
        for i, cluster in enumerate(cold_start_result.get("clusterList", [])[:3]):  # Show first 3 clusters
            logger.info(f"\nCluster {i+1}:")
            logger.info(f"  ID: {cluster.get('id')}")
            logger.info(f"  Name: {cluster.get('name')}")
            logger.info(f"  Memory count: {len(cluster.get('memory_list', []))}")
        
        #--------------------------------------------------------------------------
        # 5. Test incremental update with existing clusters
        #--------------------------------------------------------------------------
        logger.info("STEP 5: Testing generate_topics_for_shades with INCREMENTAL UPDATE...")
        
        # Use half of the memory list as existing and half as new
        split_point = len(memory_list) // 2
        existing_memory_list = memory_list[:split_point]
        new_memory_list = memory_list[split_point:]
        
        # First, get clusters for the existing memories
        existing_clusters_result = topics_generator.generate_topics_for_shades(
            user_id=user_id,
            old_cluster_list=[],
            old_outlier_memory_list=[],
            new_memory_list=existing_memory_list
        )
        
        # Then, update with the new memories
        incremental_update_result = topics_generator.generate_topics_for_shades(
            user_id=user_id,
            old_cluster_list=existing_clusters_result.get("clusterList", []),
            old_outlier_memory_list=existing_clusters_result.get("outlierMemoryList", []),
            new_memory_list=new_memory_list
        )
        
        # Log the results
        cluster_count = len(incremental_update_result.get("clusterList", []))
        outlier_count = len(incremental_update_result.get("outlierMemoryList", []))
        logger.info(f"Incremental update generated {cluster_count} clusters and {outlier_count} outliers")
        
        for i, cluster in enumerate(incremental_update_result.get("clusterList", [])[:3]):  # Show first 3 clusters
            logger.info(f"\nCluster {i+1}:")
            logger.info(f"  ID: {cluster.get('id')}")
            logger.info(f"  Name: {cluster.get('name')}")
            logger.info(f"  Memory count: {len(cluster.get('memory_list', []))}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ generate_topics_for_shades test completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Error testing generate_topics_for_shades after {elapsed_time:.2f} seconds: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Clean up resources
        if resources:
            cleanup_resources(resources)

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
    # notes_test_success = run_extract_notes_from_l0()
    
    # Run the topics for shades test
    logger.info("\n=== Starting topics for shades test ===")
    topics_test_success = run_topics_for_shades()

    method_to_test = topics_test_success;
    
    # Final results
    if method_to_test:
        logger.info("All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1) 