"""
L1Manager for orchestrating the L1 generation process.

This module provides the L1Manager class that orchestrates the process of generating
L1 level knowledge representations from L0 data. It coordinates between different
generators and manages data flow between components.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from app.models.l1.generation_result import L1GenerationResult
from app.models.l1.bio import Bio
from app.models.l1.note import Note, Chunk
from app.models.l1.db_models import L1Version, L1GlobalBiography, L1StatusBiography
from app.providers.l1.postgres_adapter import PostgresAdapter
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter
from app.providers.l1.weaviate_adapter import WeaviateAdapter

# Import generator components
from app.processors.l1.l1_generator import L1Generator
from app.processors.l1.topics_generator import TopicsGenerator
from app.processors.l1.shade_generator import ShadeGenerator
from app.processors.l1.shade_merger import ShadeMerger
from app.processors.l1.biography_generator import BiographyGenerator

logger = logging.getLogger(__name__)

class L1Manager:
    """
    Manages the generation and storage of L1 level knowledge representations.
    
    This class is designed to be compatible with lpm_kernel's l1_manager.py.
    The methods and data structures maintain compatibility with the original code,
    including parameter names and return types to ensure seamless integration.
    The key methods _extract_notes_from_l0 and generate_l1_from_l0 correspond to
    extract_notes_from_documents and generate_l1_from_l0 in lpm_kernel respectively.
    
    Attributes:
        postgres_adapter: Adapter for PostgreSQL database operations
        wasabi_adapter: Adapter for Wasabi storage operations
        weaviate_adapter: Adapter for Weaviate vector database operations
        l1_generator: Core logic for generating L1 representations
    """
    
    def __init__(
        self,
        postgres_adapter: PostgresAdapter,
        wasabi_adapter: WasabiStorageAdapter,
        weaviate_adapter: WeaviateAdapter,
        l1_generator: L1Generator,
        topics_generator: TopicsGenerator,
        shade_generator: ShadeGenerator,
        shade_merger: ShadeMerger,
        biography_generator: BiographyGenerator
    ):
        """
        Initialize the L1Manager.
        
        Args:
            postgres_adapter: PostgresAdapter instance for database operations
            wasabi_adapter: WasabiStorageAdapter instance for S3 operations
            weaviate_adapter: WeaviateAdapter instance for vector operations
            l1_generator: L1Generator instance for generating L1 representations
            topics_generator: TopicsGenerator instance for topics/clusters generation
            shade_generator: ShadeGenerator instance for shade generation
            shade_merger: ShadeMerger instance for merging similar shades
            biography_generator: BiographyGenerator instance for biography generation
        """
        self.postgres_adapter = postgres_adapter
        self.wasabi_adapter = wasabi_adapter
        self.weaviate_adapter = weaviate_adapter
        self.l1_generator = l1_generator
        
        # Store specialized generator components
        self.topics_generator = topics_generator
        self.shade_generator = shade_generator
        self.shade_merger = shade_merger
        self.biography_generator = biography_generator
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
    
    def generate_l1_from_l0(self, user_id: str) -> L1GenerationResult:
        """
        Generate L1 level knowledge representation from L0 data.
        
        Args:
            user_id: The user ID to generate L1 data for
            
        Returns:
            L1GenerationResult: The generated L1 data or None if there was an error
        """
        self.logger.info(f"Starting L1 generation for user {user_id}")
        
        try:
            # 1. Extract notes and memories from L0 data
            notes_list, memory_list = self._extract_notes_from_l0(user_id)
            
            if not notes_list or not memory_list:
                self.logger.error(f"No valid documents found for processing for user {user_id}")
                return None
                
            # 2. Generate L1 data
            
            # 2.1 Generate topic clusters for shades
            self.logger.info("Generating clusters...")
            clusters = self.topics_generator.generate_topics_for_shades(
                user_id=user_id,
                old_cluster_list=[], 
                old_outlier_memory_list=[], 
                new_memory_list=memory_list
            )
            self.logger.info(f"Generated clusters: {bool(clusters)}")
            
            # 2.2 Generate chunk topics
            self.logger.info("Generating chunk topics...")
            chunk_topics = self.topics_generator.generate_topics(user_id=user_id, notes=notes_list)
            self.logger.info(f"Generated chunk topics: {bool(chunk_topics)}")
            self.logger.debug(f"Chunk topics content: {chunk_topics}")
            
            # 2.3 Generate shades for each cluster and merge them
            shades = []
            if clusters and "clusterList" in clusters:
                for cluster in clusters.get("clusterList", []):
                    # Use memoryId for compatibility with lpm_kernel
                    cluster_memory_ids = [
                        str(m.get("memoryId")) for m in cluster.get("memoryList", [])
                    ]
                    self.logger.info(f"Processing cluster with {len(cluster_memory_ids)} memories")
                    
                    # Find notes with matching IDs, using Note.id or Note.noteId
                    cluster_notes = [
                        note for note in notes_list if str(note.id) in cluster_memory_ids
                    ]
                    
                    if cluster_notes:
                        self.logger.info(f"Generating shade for cluster with {len(cluster_notes)} notes")
                        shade = self.shade_generator.generate_shade(
                            user_id=user_id,
                            old_memory_list=[],
                            new_memory_list=cluster_notes,
                            shade_info_list=[]
                        )
                        
                        if shade:
                            shades.append(shade)
                            self.logger.info(f"Generated shade: {shade.name if hasattr(shade, 'name') else 'Unknown'}")
            
            self.logger.info(f"Generated {len(shades)} shades")
            
            # 2.4 Merge similar shades
            merged_shades_result = self.shade_merger.merge_shades(user_id=user_id, shades=shades)
            self.logger.info(f"Merged shades success: {merged_shades_result.success}")
            self.logger.info(
                f"Number of merged shades: {len(merged_shades_result.merge_shade_list) if merged_shades_result.success else 0}"
            )
            
            # 2.5 Generate global biography
            self.logger.info("Generating global biography...")
            # Use shades_list for compatibility with lpm_kernel Bio class
            bio = self.biography_generator.generate_global_biography(
                user_id=user_id,
                old_profile=Bio(
                    shades_list=merged_shades_result.merge_shade_list
                    if merged_shades_result.success
                    else []
                ),
                cluster_list=clusters.get("clusterList", []),
            )
            self.logger.info(f"Generated global biography successfully")
            
            # 3. Store L1 data
            self._store_l1_data(
                user_id=user_id,
                bio=bio,
                clusters=clusters,
                chunk_topics=chunk_topics,
                shades=merged_shades_result.merge_shade_list if merged_shades_result.success else []
            )
            
            # 4. Build result object
            result = L1GenerationResult(
                bio=bio, 
                clusters=clusters, 
                chunk_topics=chunk_topics
            )
            
            self.logger.info(f"L1 generation completed successfully for user {user_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in L1 generation for user {user_id}: {str(e)}", exc_info=True)
            raise
    
    def _extract_notes_from_l0(self, user_id: str) -> Tuple[List[Note], List[Dict[str, Any]]]:
        """
        Extract notes and memory list from L0 data.
        
        Args:
            user_id: The user ID to extract notes for
            
        Returns:
            Tuple of (notes_list, memory_list)
        """
        notes_list = []
        memory_list = []
        
        # Get documents with L0 data from PostgreSQL
        documents = self.postgres_adapter.get_documents_with_l0(user_id)
        self.logger.info(f"Found {len(documents)} documents with L0 data for user {user_id}")
        
        for doc in documents:
            doc_id = doc.id
            
            # Get document embedding from Weaviate
            doc_embedding = self.weaviate_adapter.get_document_embedding(user_id, doc_id)
            
            # Process document embedding - handle potential dict format from Weaviate
            if doc_embedding:
                if isinstance(doc_embedding, dict) and 'default' in doc_embedding:
                    self.logger.info(f"Document {doc_id} embedding is in dict format with 'default' key, extracting vector")
                    doc_embedding = doc_embedding['default']
            
            # Get document chunks from Weaviate
            chunks = self.weaviate_adapter.get_document_chunks(user_id, doc_id)
            
            # Get chunk embeddings from Weaviate
            all_chunk_embeddings = self.weaviate_adapter.get_chunk_embeddings_by_document(user_id, doc_id)
            
            # Process chunk embeddings - handle potential dict format from Weaviate
            processed_chunk_embeddings = {}
            for chunk_id, embedding in all_chunk_embeddings.items():
                if isinstance(embedding, dict) and 'default' in embedding:
                    self.logger.info(f"Chunk {chunk_id} embedding is in dict format with 'default' key, extracting vector")
                    processed_chunk_embeddings[chunk_id] = embedding['default']
                else:
                    processed_chunk_embeddings[chunk_id] = embedding
            
            # Use the processed embeddings
            all_chunk_embeddings = processed_chunk_embeddings
            
            # Skip documents with missing data
            if not doc_embedding:
                self.logger.warning(f"Document {doc_id} missing document embedding")
                continue
            if not chunks:
                self.logger.warning(f"Document {doc_id} missing chunks")
                continue
            if not all_chunk_embeddings:
                self.logger.warning(f"Document {doc_id} missing chunk embeddings")
                continue
            
            # Ensure create_time is in string format
            # Use uploaded_at instead of create_time (which doesn't exist in our Document model)
            create_time = doc.uploaded_at if hasattr(doc, 'uploaded_at') else datetime.now()
            if isinstance(create_time, datetime):
                create_time = create_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get document insight, summary, and raw content from Wasabi
            document_data = self.wasabi_adapter.get_document(user_id, doc_id)
            
            # Handle case where document_data is None
            if document_data is None:
                self.logger.warning(f"Document {doc_id} missing document data from Wasabi")
                document_data = {}
                
            insight_data = document_data.get("insight", {}) or {}
            summary_data = document_data.get("summary", {}) or {}
            
            # Build Note object with parameter names matching lpm_kernel
            note = Note(
                id=doc_id,  # This field will be mapped to 'id' in the Note object
                content=document_data.get("raw_content", ""),
                create_time=create_time,  # This will be mapped to 'create_time'
                embedding=np.array(doc_embedding),
                chunks=[
                    Chunk(
                        id=chunk.id,
                        document_id=doc_id,
                        content=chunk.content,
                        embedding=np.array(all_chunk_embeddings.get(chunk.id))
                        if all_chunk_embeddings.get(chunk.id)
                        else None,
                        tags=getattr(chunk, "tags", None),
                        topic=getattr(chunk, "topic", None)
                    )
                    for chunk in chunks
                    if all_chunk_embeddings.get(chunk.id)
                ],
                title=insight_data.get("title", ""),
                summary=summary_data.get("summary", ""),
                insight=insight_data.get("insight", ""),
                tags=summary_data.get("keywords", []),
                memory_type="TEXT"  # This will be mapped to 'memory_type'
            )
            notes_list.append(note)
            
            # Add to memory list for clustering - matching lpm_kernel format
            # Make sure we're using a properly processed embedding, not a dictionary
            processed_embedding = doc_embedding
            if isinstance(processed_embedding, dict) and 'default' in processed_embedding:
                processed_embedding = processed_embedding['default']
                
            memory_list.append({
                "memoryId": str(doc_id),
                "embedding": processed_embedding
            })
        
        self.logger.info(f"Extracted {len(notes_list)} notes for user {user_id}")
        return notes_list, memory_list
    
    def _store_l1_data(
        self,
        user_id: str,
        bio: Bio,
        clusters: Dict[str, Any],
        chunk_topics: Dict[str, Dict],
        shades: List[Dict[str, Any]]
    ) -> None:
        """
        Store L1 data in PostgreSQL, Wasabi, and Weaviate.
        
        Args:
            user_id: User ID
            bio: Biography data
            clusters: Clusters data
            chunk_topics: Chunk topics data (now a Dict[str, Dict])
            shades: Shades data
        """
        try:
            # 1. Create a new version record for tracking
            latest_version = self.postgres_adapter.get_latest_version(user_id)
            new_version = (latest_version or 0) + 1
            self.postgres_adapter.create_version(user_id, new_version, "processing")
            
            # 2. Store topics/clusters in Weaviate and PostgreSQL
            if clusters and "clusterList" in clusters:
                for cluster in clusters.get("clusterList", []):
                    # Get cluster data
                    cluster_id = cluster.get("clusterId")
                    cluster_name = cluster.get("clusterName")
                    center_embedding = cluster.get("centerEmbed")
                    document_ids = [
                        str(m.get("memoryId")) for m in cluster.get("memoryList", [])
                    ]
                    
                    # Store in Weaviate (vector DB)
                    self.weaviate_adapter.store_cluster(
                        user_id=user_id,
                        cluster_id=cluster_id,
                        name=cluster_name,
                        center_embedding=center_embedding,
                        version=new_version
                    )
                    
                    # Store in PostgreSQL (metadata DB)
                    self.postgres_adapter.store_cluster(
                        user_id=user_id,
                        cluster_id=cluster_id,
                        name=cluster_name,
                        document_ids=document_ids,
                        version=new_version
                    )
            
            # 3. Store chunk topics in PostgreSQL
            if chunk_topics:
                self.postgres_adapter.store_chunk_topics(
                    user_id=user_id,
                    chunk_topics=chunk_topics,
                    version=new_version
                )
            
            # 4. Store shades in PostgreSQL and Wasabi
            if shades:
                for shade in shades:
                    shade_id = shade.get("id")
                    # Store in Wasabi (object storage)
                    self.wasabi_adapter.store_shade(
                        user_id=user_id,
                        shade_id=shade_id,
                        shade_data=shade,
                        version=new_version
                    )
                    
                    # Store metadata in PostgreSQL
                    self.postgres_adapter.store_shade(
                        user_id=user_id,
                        shade_id=shade_id,
                        name=shade.get("name"),
                        summary=shade.get("summary"),
                        confidence=shade.get("confidence"),
                        source_clusters=shade.get("source_clusters", []),
                        version=new_version
                    )
            
            # 5. Store biography in PostgreSQL and Wasabi
            # Store the global biography in Wasabi (full content)
            bio_id = f"global_{new_version}"
            self.wasabi_adapter.store_biography(
                user_id=user_id,
                bio_id=bio_id,
                bio_data={
                    "content_first_view": bio.content_first_view,
                    "content_second_view": bio.content_second_view,
                    "content_third_view": bio.content_third_view,
                    "summary_first_view": bio.summary_first_view,
                    "summary_second_view": bio.summary_second_view,
                    "summary_third_view": bio.summary_third_view
                },
                version=new_version
            )
            
            # Store global biography metadata in PostgreSQL
            self.postgres_adapter.store_global_biography(
                user_id=user_id,
                content=bio.content_second_view,
                content_third_view=bio.content_third_view,
                summary=bio.summary_second_view,
                summary_third_view=bio.summary_third_view,
                version=new_version
            )
            
            # 6. Update version status to completed
            self.postgres_adapter.update_version_status(
                user_id=user_id,
                version=new_version,
                status="completed"
            )
            
            self.logger.info(f"Successfully stored L1 data for user {user_id}, version {new_version}")
            
        except Exception as e:
            self.logger.error(f"Error storing L1 data for user {user_id}: {str(e)}", exc_info=True)
            
            # Update version status to failed if created
            if new_version:
                self.postgres_adapter.update_version_status(
                    user_id=user_id,
                    version=new_version,
                    status="failed",
                    error_message=str(e)
                )
            
            raise
    
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