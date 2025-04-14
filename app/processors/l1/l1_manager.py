"""
L1Manager for orchestrating the L1 generation process.

This module provides the L1Manager class that orchestrates the process of generating
L1 level knowledge representations from L0 data. It coordinates between different
generators and manages data flow between components.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.models.l1.generation_result import L1GenerationResult
from app.models.l1.bio import Bio
from app.models.l1.note import Note, Chunk
from app.models.l1.db_models import L1Version, L1GlobalBiography, L1StatusBiography
from app.providers.l1.postgres_adapter import PostgresAdapter
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter
from app.providers.l1.weaviate_adapter import WeaviateAdapter

# Will be implemented
from app.processors.l1.l1_generator import L1Generator
from app.processors.l1.topics_generator import TopicsGenerator
from app.processors.l1.shade_generator import ShadeGenerator
from app.processors.l1.biography_generator import BiographyGenerator

logger = logging.getLogger(__name__)

class L1Manager:
    """
    Orchestrates the L1 generation process.
    
    This class coordinates between different generators (L1Generator, TopicsGenerator,
    ShadeGenerator, BiographyGenerator) and manages data flow between components.
    It handles extracting notes from L0, generating topics and clusters,
    generating shades, merging shades, and generating biographies.
    
    Attributes:
        postgres_adapter: Adapter for PostgreSQL database operations
        wasabi_adapter: Adapter for Wasabi storage operations
        weaviate_adapter: Adapter for Weaviate vector database operations
        l1_generator: Core logic for generating L1 representations
    """
    
    def __init__(
        self,
        postgres_adapter: Optional[PostgresAdapter] = None,
        wasabi_adapter: Optional[WasabiStorageAdapter] = None,
        weaviate_adapter: Optional[WeaviateAdapter] = None,
        l1_generator: Optional[L1Generator] = None
    ):
        """
        Initialize the L1Manager.
        
        Args:
            postgres_adapter: PostgresAdapter instance for database operations
            wasabi_adapter: WasabiStorageAdapter instance for S3 operations
            weaviate_adapter: WeaviateAdapter instance for vector operations
            l1_generator: L1Generator instance for generating L1 representations
        """
        self.postgres_adapter = postgres_adapter or PostgresAdapter()
        self.wasabi_adapter = wasabi_adapter or WasabiStorageAdapter()
        self.weaviate_adapter = weaviate_adapter or WeaviateAdapter()
        self.l1_generator = l1_generator or L1Generator()
    
    def generate_l1_from_l0(self, user_id: str) -> L1GenerationResult:
        """
        Generate L1 level knowledge representation from L0 data.
        
        Args:
            user_id: The user ID to generate L1 data for
            
        Returns:
            L1GenerationResult containing the generated data and status
        """
        try:
            # 1. Create a new version record
            latest_version = self.postgres_adapter.get_latest_version(user_id)
            new_version = (latest_version or 0) + 1
            version_record = self.postgres_adapter.create_version(user_id, new_version)
            
            # 2. Extract notes and memories from L0
            notes_list, memory_list = self._extract_notes_from_l0(user_id)
            
            if not notes_list or len(notes_list) == 0:
                error_msg = "No valid documents found for processing"
                logger.error(error_msg)
                self.postgres_adapter.update_version_status(
                    user_id, new_version, "failed", error_msg
                )
                return L1GenerationResult.failure(error_msg)
            
            # 3. Generate L1 data
            try:
                # 3.1 Generate topics and clusters
                clusters = self.l1_generator.gen_topics_for_shades(
                    user_id=user_id,
                    old_cluster_list=[],
                    old_outlier_memory_list=[],
                    new_memory_list=memory_list
                )
                logger.info(f"Generated clusters: {bool(clusters)}")
                
                # 3.2 Generate chunk topics
                chunk_topics = self.l1_generator.generate_topics(notes_list)
                logger.info(f"Generated chunk topics: {bool(chunk_topics)}")
                
                # 3.3 Generate shades for each cluster and merge them
                shades = []
                if clusters and "clusterList" in clusters:
                    for cluster in clusters.get("clusterList", []):
                        cluster_memory_ids = [
                            str(m.get("memoryId")) for m in cluster.get("memoryList", [])
                        ]
                        logger.info(
                            f"Processing cluster with {len(cluster_memory_ids)} memories"
                        )
                        
                        cluster_notes = [
                            note for note in notes_list if str(note.id) in cluster_memory_ids
                        ]
                        if cluster_notes:
                            shade = self.l1_generator.gen_shade_for_cluster(
                                user_id=user_id,
                                old_shades=[],
                                cluster_notes=cluster_notes,
                                memory_list=[]
                            )
                            if shade:
                                shades.append(shade)
                                logger.info(f"Generated shade: {shade.name if hasattr(shade, 'name') else 'Unknown'}")
                
                logger.info(f"Generated {len(shades)} shades")
                merged_shades = self.l1_generator.merge_shades(user_id, shades)
                logger.info(f"Merged shades success: {merged_shades.success}")
                logger.info(
                    f"Number of merged shades: {len(merged_shades.merge_shade_list) if merged_shades.success else 0}"
                )
                
                # 3.4 Generate global biography
                bio = self.l1_generator.gen_global_biography(
                    user_id=user_id,
                    old_profile=Bio(
                        shades_list=merged_shades.merge_shade_list if merged_shades.success else []
                    ),
                    cluster_list=clusters.get("clusterList", []),
                )
                logger.info(f"Generated global biography")
                
                # 4. Store the results in PostgreSQL/Wasabi/Weaviate
                self._store_l1_data(user_id, new_version, bio, clusters, chunk_topics)
                
                # 5. Update version status to completed
                self.postgres_adapter.update_version_status(
                    user_id, new_version, "completed"
                )
                
                # 6. Build result object with updated structure
                result = L1GenerationResult.success(
                    bio=bio, 
                    clusters=clusters, 
                    chunk_topics=chunk_topics
                )
                
                logger.info("L1 generation completed successfully")
                return result
                
            except Exception as e:
                error_msg = f"Error in L1 generation: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.postgres_adapter.update_version_status(
                    user_id, new_version, "failed", error_msg
                )
                return L1GenerationResult.failure(error_msg)
                
        except Exception as e:
            error_msg = f"Error in L1 generation process: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return L1GenerationResult.failure(error_msg)
    
    def _extract_notes_from_l0(self, user_id: str) -> Tuple[List[Note], List[Dict[str, Any]]]:
        """
        Extract notes and memory list from L0 data.
        
        Args:
            user_id: The user ID to extract notes for
            
        Returns:
            Tuple of (notes_list, memory_list)
        """
        # TODO: Implement extracting Notes from L0 documents
        # This will need to query documents with L0 data from PostgreSQL
        # and transform them into Note objects
        
        # Placeholder implementation
        return [], []
    
    def _store_l1_data(
        self,
        user_id: str,
        version: int,
        bio: Bio,
        clusters: Dict[str, Any],
        chunk_topics: Dict[str, Dict]  # Updated from List[Dict[str, Any]]
    ) -> None:
        """
        Store L1 data in PostgreSQL, Wasabi, and Weaviate.
        
        Args:
            user_id: User ID
            version: Version number
            bio: Biography data
            clusters: Clusters data
            chunk_topics: Chunk topics data (now a Dict[str, Dict])
        """
        # TODO: Implement storing L1 data in PostgreSQL, Wasabi, and Weaviate
        pass
    
    def get_latest_global_bio(self, user_id: str) -> Optional[Bio]:
        """
        Get the latest global biography for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            The latest global biography or None if not found
        """
        # TODO: Implement retrieving the latest global biography
        return None
    
    def get_latest_status_bio(self, user_id: str) -> Optional[Bio]:
        """
        Get the latest status biography for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            The latest status biography or None if not found
        """
        # TODO: Implement retrieving the latest status biography
        return None
    
    def get_latest_l1_version(self, user_id: str) -> Optional[L1Version]:
        """
        Get the latest L1 version for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            The latest L1Version or None if not found
        """
        return self.postgres_adapter.get_latest_version(user_id) 