"""
TopicsGenerator for creating topic clusters from document embeddings.

This module provides the TopicsGenerator class that creates topic clusters
from document embeddings using clustering algorithms and generates topic labels
for each cluster.
"""
import logging
import json
import copy
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from collections import deque, defaultdict
import math
import itertools
import uuid

from app.models.l1.note import Note, Chunk
from app.models.l1.topic import Memory, Cluster  # Import our Cluster and Memory classes
from app.services.llm_service import LLMService
from app.processors.l1.utils import find_connected_components  # Import the utility function

logger = logging.getLogger(__name__)

# LLM Prompt Templates
SYS_TOPICS = """You are a skilled wordsmith with extensive experience in managing structured knowledge documents. Given a knowledge chunk, your main task involves crafting phrases that accurately represent provided chunk as "topics" and generating concise "tags" for categorization purposes. The tags, several nouns, should be broader and more general than the topic. Here are some examples illustrating effective pairing of topics and tags:

{"topic": "Decoder-only transformers pretraining on large-scale corpora", "tags": ["Transformers", "Pretraining", "Large-scale corpora"]}
{"topic": "Formula 1 racing car aerodynamics learning", "tags": ["Formula 1", "Racing", "Aerodynamics"]}
{"topic": "1980s Progressive Rock bands and their discographies", "tags": ["Progressive Rock", "Bands", "Discographies"]}
{"topic": "Czech Republic's history and culture during medieval times", "tags": ["Czech Republic", "History", "Culture"]}
{"topic": "Revolution of European Political Economy in the 19th century", "tags": ["Political Economy", "Revolution", "Europe"]}

Guidelines for generating effective "topics" and "tags" are as follows:
1. A good topic should be concise, informative, and specifically capture the essence of the note without being overly broad or vague.
2. The tags should be 3-5 nouns and more general than the topic, serving as a category or a prompt for further dialogue.
3. Ideally, a topic should comprise 5-10 words, while each tag should be limited to 1-3 words.
4. Use double quotes in your response and make sure it can be parsed using json.loads(), as shown in the examples above."""

USR_TOPICS = """Please generate a topic and tags for the knowledge chunk provided below, using the format of the examples previously mentioned. Just produce the topic and tags using the same JSON format as the examples.

{chunk}
"""

SYS_COMB = """You are a skilled wordsmith with extensive experience in managing structured knowledge documents. Given a set of topics and a set of tags, your main task involves crafting a new topic and a new set of tags that accurately represent the provided topics and tags. Here are some examples illustrating effective merging of topics and tags:
1. Given topics: "Decoder-only transformers pretraining on large-scale corpora", "Parameter Effcient LLM Finetuning" and tags: ["Transformers", "Pretraining", "Large-scale corpora"], ["LLM", "Parameter Efficient", Finetuning"], you can merge them into: {"topic": "Efficient transformers pretraining and finetuning on large-scale corpora", "tags": ["Transformers", "Pretraining", "Finetuning"]}.
2. Given topics: "Formula 1 racing car aerodynamics learning", "Formula 1 racing car design optimization" and tags: ["Formula 1", "Racing", "Aerodynamics"], ["Formula 1", "Design", "Optimization"], you can merge them into: {"topic": "Formula 1 racing car aerodynamics and design optimization", "tags": ["Formula 1", "Racing", "Aerodynamics", "Design", "Optimization"]}.

Guidelines for generating representative topic and tags are as follows:
1. The new topic should be a concise and informative summary of the provided topics, capturing the essence of the topics without being overly broad or vague.
2. The new tags should be 3-5 nouns, combining the tags from the provided topics, and should be more general than the new topic, serving as a category or a prompt for further dialogue.
3. Ideally, a topic should comprise 5-10 words, while each tag should be limited to 1-3 words.
4. Use double quotes in your response and make sure it can be parsed using json.loads(), as shown in the examples above."""

USR_COMB = """Please generate the new topic and new tags for the given set of topics and tags, using the format of the examples previously mentioned. Just produce the new topic and tags using the same JSON format as the examples.

Topics: {topics}

Tags list: {tags}
"""


class TopicsGenerator:
    """
    Creates topic clusters from document embeddings.
    
    This class implements clustering algorithms to group similar documents
    and generates topic labels for each cluster. It handles outlier documents
    and provides methods for cold-start clustering and incremental updates.
    
    Attributes:
        llm_service: Service for LLM interactions
        default_cophenetic_distance: Default distance threshold for hierarchical clustering
        default_outlier_cutoff_distance: Default distance threshold to determine outliers
        default_cluster_merge_distance: Default distance threshold for merging clusters
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        default_cophenetic_distance: float = 1.0,
        default_outlier_cutoff_distance: float = 0.9,
        default_cluster_merge_distance: float = 0.5
    ):
        """
        Initialize the TopicsGenerator.
        
        Args:
            llm_service: Service for LLM interactions
            default_cophenetic_distance: Default distance threshold for hierarchical clustering
            default_outlier_cutoff_distance: Default distance threshold to determine outliers
            default_cluster_merge_distance: Default distance threshold for merging clusters
        """
        self.llm_service = llm_service
        self.default_cophenetic_distance = default_cophenetic_distance
        self.default_outlier_cutoff_distance = default_outlier_cutoff_distance
        self.default_cluster_merge_distance = default_cluster_merge_distance
    
    def generate_topics(self, notes_list: List[Note]) -> Dict[str, Any]:
        """
        Generate topics from a list of notes.
        
        Args:
            notes_list: List of Note objects to process
            
        Returns:
            A dictionary containing topic data
        """
        logger.info(f"Processing {len(notes_list)} notes for topic generation")
        
        # Log sample of notes for debugging
        for i, note in enumerate(notes_list[:3]):  # Only log first 3 notes
            logger.info(f"\nNote {i + 1}:")
            logger.info(f"  ID: {note.id}")
            logger.info(f"  Title: {note.title}")
            logger.info(f"  Content: {note.content[:200]}...")  # only showing first 200 characters
            logger.info(f"  Number of chunks: {len(note.chunks)}")
        
        # Perform cold start clustering
        topics_data = self._cold_start(notes_list)
        
        return topics_data
    
    def generate_topics_for_shades(
        self,
        user_id: str,  # Add user_id parameter for consistency
        old_cluster_list: List[Dict[str, Any]],
        old_outlier_memory_list: List[Dict[str, Any]],
        new_memory_list: List[Dict[str, Any]],
        cophenetic_distance: Optional[float] = None,
        outlier_cutoff_distance: Optional[float] = None,
        cluster_merge_distance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate topic clusters for shades by updating existing clusters or creating new ones.
        
        Args:
            user_id: User ID for logging
            old_cluster_list: List of existing clusters
            old_outlier_memory_list: List of outlier memories from previous run
            new_memory_list: List of new memories to process
            cophenetic_distance: Distance threshold for hierarchical clustering
            outlier_cutoff_distance: Distance threshold to determine outliers
            cluster_merge_distance: Distance threshold for merging clusters
            
        Returns:
            A dictionary containing updated cluster list and outlier memory list
        """
        # Log input data
        # logger.info(f"DIAGNOSTIC INPUT: User ID: {user_id}")
        # logger.info(f"DIAGNOSTIC INPUT: Old clusters count: {len(old_cluster_list)}")
        # logger.info(f"DIAGNOSTIC INPUT: Old outliers count: {len(old_outlier_memory_list)}")
        # logger.info(f"DIAGNOSTIC INPUT: New memories count: {len(new_memory_list)}")
        
        # Check a sample of the memory structure
        if new_memory_list:
            sample_memory = new_memory_list[0]
            # logger.info(f"DIAGNOSTIC INPUT: Sample memory keys: {list(sample_memory.keys())}")
            if "memoryId" in sample_memory:
                # logger.info(f"DIAGNOSTIC INPUT: Sample memory ID: {sample_memory['memoryId']}")
                pass
            if "embedding" in sample_memory:
                embedding = sample_memory["embedding"]
                if isinstance(embedding, list):
                    # logger.info(f"DIAGNOSTIC INPUT: Sample embedding type: list, length: {len(embedding)}")
                    pass
                elif isinstance(embedding, np.ndarray):
                    # logger.info(f"DIAGNOSTIC INPUT: Sample embedding type: ndarray, shape: {embedding.shape}")
                    pass
                else:
                    # logger.info(f"DIAGNOSTIC INPUT: Sample embedding type: {type(embedding)}")
                    pass
        
        # Use default values if not provided
        cophenetic_distance = cophenetic_distance or self.default_cophenetic_distance
        outlier_cutoff_distance = outlier_cutoff_distance or self.default_outlier_cutoff_distance
        cluster_merge_distance = cluster_merge_distance or self.default_cluster_merge_distance
        
        # logger.info(f"DIAGNOSTIC PARAMS: cophenetic_distance={cophenetic_distance}, outlier_cutoff_distance={outlier_cutoff_distance}, cluster_merge_distance={cluster_merge_distance}")
        
        logger.info(f"Generating topics for shades with {len(new_memory_list)} new memories for user {user_id}")
        
        # Convert memory dicts to Memory objects
        new_memory_list = [self._convert_to_memory_object(memory) for memory in new_memory_list]
        new_memory_list = [memory for memory in new_memory_list if memory.embedding is not None]
        
        # Convert cluster dicts to Cluster objects
        old_cluster_list = [self._convert_to_cluster_object(cluster) for cluster in old_cluster_list]
        old_outlier_memory_list = [self._convert_to_memory_object(memory) for memory in old_outlier_memory_list]
        
        # If no existing clusters, perform cold start
        if not old_cluster_list:
            logger.info("No existing clusters, performing initial strategy")
            # Initial strategy for clustering
            cluster_list, outlier_memory_list = self._clusters_initial_strategy(
                new_memory_list, cophenetic_distance
            )
        else:
            # Update strategy for existing clusters
            logger.info("Updating existing clusters")
            cluster_list, outlier_memory_list = self._clusters_update_strategy(
                old_cluster_list,
                old_outlier_memory_list,
                new_memory_list,
                cophenetic_distance,
                outlier_cutoff_distance,
                cluster_merge_distance
            )
        
        logger.info(f"Generated {len(cluster_list)} clusters")
        logger.info(f"In-cluster memory count: {sum([len(cluster.memory_list) for cluster in cluster_list])}")
        logger.info(f"Outlier memory count: {len(outlier_memory_list)}")
        
        # Convert to the expected output format
        return {
            "clusterList": [cluster.to_json() for cluster in cluster_list],
            "outlierMemoryList": [memory.to_json() for memory in outlier_memory_list]
        }
    
    def _convert_to_memory_object(self, memory_data: Dict[str, Any]) -> Memory:
        """
        Convert memory dictionary to Memory object.
        
        Args:
            memory_data: Dictionary containing memory data
            
        Returns:
            Memory object
        """
        memory_id = memory_data.get("memoryId", "")
        embedding = memory_data.get("embedding", [])
        metadata = {k: v for k, v in memory_data.items() if k not in ["memoryId", "embedding"]}
        
        # Diagnostic check for embedding shape
        if isinstance(embedding, list) and embedding:
            # logger.debug(f"DIAGNOSTIC: Memory {memory_id} has embedding list of length {len(embedding)}")
            pass
        elif isinstance(embedding, np.ndarray):
            # logger.debug(f"DIAGNOSTIC: Memory {memory_id} has numpy embedding of shape {embedding.shape}")
            pass
        elif embedding is None:
            # logger.warning(f"DIAGNOSTIC: Memory {memory_id} has None embedding")
            pass
        else:
            # logger.warning(f"DIAGNOSTIC: Memory {memory_id} has unexpected embedding type: {type(embedding)}")
            pass
        
        memory = Memory(
            memory_id=memory_id,
            embedding=embedding,
            metadata=metadata
        )
        
        return memory
    
    def _convert_to_cluster_object(self, cluster_data: Dict[str, Any]) -> Cluster:
        """
        Convert cluster dictionary to Cluster object.
        
        Args:
            cluster_data: Dictionary containing cluster data
            
        Returns:
            Cluster object
        """
        # Log full input cluster data for debugging
        # logger.info(f"DIAGNOSTIC: Converting cluster data keys: {list(cluster_data.keys())}")
        
        # Extract clusterId
        cluster_id = cluster_data.get("clusterId", "")
        if not cluster_id:
            # logger.warning(f"DIAGNOSTIC: Missing clusterId in cluster data")
            # logger.warning(f"DIAGNOSTIC: Full cluster data: {cluster_data}")
            # Try to provide a fallback ID
            cluster_id = str(uuid.uuid4())[:8]
            # logger.warning(f"DIAGNOSTIC: Generated fallback ID: {cluster_id}")
        else:
            # logger.info(f"DIAGNOSTIC: Found clusterId: {cluster_id}")
            pass
        
        # Extract memory list
        memory_list = []
        memory_list_data = cluster_data.get("memoryList", [])
        # logger.info(f"DIAGNOSTIC: Found {len(memory_list_data)} memories in memory list")
        
        # Convert each memory in the memory list
        for memory_data in memory_list_data:
            try:
                memory = self._convert_to_memory_object(memory_data)
                memory_list.append(memory)
            except Exception as e:
                # logger.error(f"DIAGNOSTIC: Error converting memory: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Get other properties
        name = cluster_data.get("topic", "Unknown Topic")
        tags = cluster_data.get("tags", [])
        metadata = {"tags": tags}
        
        # Add any other metadata
        if "mergeList" in cluster_data:
            metadata["merge_list"] = cluster_data.get("mergeList", [])
        
        # Check for center embedding
        center_embedding = cluster_data.get("centerEmbedding", None)
        if center_embedding is not None:
            # logger.info(f"DIAGNOSTIC: Found center embedding of type {type(center_embedding)}")
            pass
        
        # logger.info(f"DIAGNOSTIC: Created cluster {cluster_id} with {len(memory_list)} memories and name: {name}")
        
        # Create the cluster object
        cluster = Cluster(
            id=cluster_id,
            name=name,
            memory_list=memory_list,
            metadata=metadata,
            center_embedding=center_embedding
        )
        
        return cluster
    
    def _clusters_initial_strategy(
        self,
        memory_list: List[Memory],
        cophenetic_distance: float,
        size_threshold: int = None
    ) -> Tuple[List[Cluster], List[Memory]]:
        """
        Initial clustering strategy for memories without existing clusters.
        
        Args:
            memory_list: List of Memory objects to cluster
            cophenetic_distance: Distance threshold for hierarchical clustering
            size_threshold: Minimum size threshold for valid clusters
            
        Returns:
            A tuple containing (generated_clusters, outlier_memories)
        """
        if not memory_list:
            # logger.info("DIAGNOSTIC: memory_list is empty, returning empty clusters and outliers")
            return [], []
            
        # logger.info(f"DIAGNOSTIC: _clusters_initial_strategy called with {len(memory_list)} memories and cophenetic_distance={cophenetic_distance}")
            
        # Log memory information for debugging
        for memory in memory_list[:3]:  # Only log a few
            logger.info(f"Memory ID: {memory.memory_id}, Embedding shape: {np.array(memory.embedding).shape if memory.embedding is not None else 'None'}")
        
        # Create matrix of embeddings for clustering
        memory_embeddings = [np.array(memory.embedding) for memory in memory_list]
        # logger.info(f"DIAGNOSTIC: Created embedding matrix with {len(memory_embeddings)} embeddings")
        
        # Perform hierarchical clustering
        if len(memory_embeddings) == 1:
            # If only one memory, create a single cluster
            # logger.info("DIAGNOSTIC: Only one memory, creating single cluster")
            clusters = np.array([1])
        else:
            # Calculate linkage and get flat clusters
            try:
                # logger.info("DIAGNOSTIC: Performing hierarchical clustering")
                linked = linkage(memory_embeddings, method="ward")
                clusters = fcluster(linked, cophenetic_distance, criterion="distance")
                # logger.info(f"DIAGNOSTIC: Generated {len(np.unique(clusters))} clusters from {len(memory_embeddings)} embeddings")
            except Exception as e:
                # logger.error(f"DIAGNOSTIC: Error in clustering: {str(e)}")
                logger.error(traceback.format_exc())
                return [], memory_list  # Return all as outliers if clustering fails
        
        labels = clusters.tolist()
        # logger.info(f"DIAGNOSTIC: Cluster labels distribution: {np.bincount(clusters)}")
        
        # Map memories to clusters
        cluster_dict = {}
        for memory, label in zip(memory_list, labels):
            if label not in cluster_dict:
                # Create new cluster with label as the id
                cluster_dict[label] = Cluster(
                    id=str(label), 
                    name="New Cluster", # TODO this is unnecessary
                    metadata={"is_new": True}
                )
                # logger.info(f"DIAGNOSTIC: Created new cluster with ID {label}")
            cluster_dict[label].memory_list.append(memory)
        
        # logger.info(f"DIAGNOSTIC: Created {len(cluster_dict)} clusters from labels")
        
        # Log cluster details before filtering
        for label, cluster in list(cluster_dict.items())[:3]:
            # logger.info(f"DIAGNOSTIC: Pre-filter cluster {label} - ID: {cluster.id}, Name: {cluster.name}, Memory count: {len(cluster.memory_list)}")
            pass
        
        # Remove small clusters
        size_threshold_before = size_threshold
        cluster_list = self._remove_immature_clusters(cluster_dict, size_threshold)
        
        # logger.info(f"DIAGNOSTIC: After filtering, {len(cluster_list)} clusters remain (threshold was {size_threshold_before})")
        
        # Prune outliers from clusters
        for i, cluster in enumerate(cluster_list[:3]):
            # logger.info(f"DIAGNOSTIC: Before pruning, cluster {i+1} has {len(cluster.memory_list)} memories")
            pass
            
        for cluster in cluster_list:
            self._prune_outliers_from_cluster(cluster)
        
        # Log after pruning
        for i, cluster in enumerate(cluster_list[:3]):
            # logger.info(f"DIAGNOSTIC: After pruning, cluster {i+1} (ID {cluster.id}) has {len(cluster.memory_list)} memories")
            pass
        
        # Identify in-cluster and outlier memories
        in_cluster_memory_ids = [
            memory.memory_id 
            for cluster in cluster_list 
            for memory in cluster.memory_list
        ]
        
        outlier_memory_list = [
            memory 
            for memory in memory_list 
            if memory.memory_id not in in_cluster_memory_ids
        ]
        
        # logger.info(f"DIAGNOSTIC: Final stats - {len(cluster_list)} clusters, {len(in_cluster_memory_ids)} in-cluster memories, {len(outlier_memory_list)} outliers")
        
        # Log final clusters
        for i, cluster in enumerate(cluster_list[:3]):
            # logger.info(f"DIAGNOSTIC: Final cluster {i+1} - ID: {cluster.id}, Name: {cluster.name}, Memory count: {len(cluster.memory_list)}")
            if cluster.memory_list:
                # logger.info(f"DIAGNOSTIC: Sample memory ID in cluster: {cluster.memory_list[0].memory_id}")
                pass
        
        return cluster_list, outlier_memory_list
    
    def _remove_immature_clusters(self, cluster_dict: Dict[Any, Cluster], size_threshold: int = None) -> List[Cluster]:
        """
        Remove clusters that are too small (immature).
        
        Args:
            cluster_dict: Dictionary mapping cluster IDs to Cluster objects
            size_threshold: Size threshold below which clusters are considered immature
            
        Returns:
            List of clusters that meet the size threshold
        """
        if not cluster_dict:
            # logger.info("DIAGNOSTIC: _remove_immature_clusters called with empty cluster_dict")
            return []
            
        # Calculate size threshold if not provided
        if not size_threshold:
            max_cluster_size = max(len(cluster.memory_list) for cluster in cluster_dict.values())
            # Use a smaller threshold - square root times 0.8 instead of just square root
            # This will allow smaller clusters to be kept
            size_threshold = math.sqrt(max_cluster_size) * 0.8
            # logger.info(f"DIAGNOSTIC: Calculated size_threshold = sqrt({max_cluster_size})*0.8 = {size_threshold}")
        else:
            # logger.info(f"DIAGNOSTIC: Using provided size_threshold = {size_threshold}")
            pass
            
        # Log before filtering
        cluster_sizes = {label: len(cluster.memory_list) for label, cluster in cluster_dict.items()}
        # logger.info(f"DIAGNOSTIC: Before filtering, cluster sizes: {cluster_sizes}")
            
        # Keep only clusters that meet the size threshold
        cluster_list = [
            cluster
            for _, cluster in cluster_dict.items()
            if len(cluster.memory_list) >= size_threshold
        ]
        
        filtered_count = len(cluster_dict) - len(cluster_list)
        # logger.info(f"DIAGNOSTIC: Removed {filtered_count} immature clusters (threshold: {size_threshold})")
        
        # Log retained clusters
        retained_sizes = {getattr(cluster, 'id', f'cluster_{i}'): len(cluster.memory_list) for i, cluster in enumerate(cluster_list)}
        # logger.info(f"DIAGNOSTIC: After filtering, retained cluster sizes: {retained_sizes}")
        
        return cluster_list
    
    def _prune_outliers_from_cluster(self, cluster: Cluster) -> None:
        """
        Remove outlier memories from a cluster based on distance from center.
        
        Args:
            cluster: Cluster object to prune outliers from
        """
        if len(cluster.memory_list) <= 2:
            # Not enough memories to prune
            return
            
        # Calculate cluster center
        memory_embeddings = [np.array(memory.embedding) for memory in cluster.memory_list]
        center = np.mean(memory_embeddings, axis=0)
        
        # Calculate distances from center
        distances = [np.linalg.norm(np.array(memory.embedding) - center) for memory in cluster.memory_list]
        
        # Calculate statistics for outlier detection
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 2 * std_dist  # Example threshold: 2 standard deviations from mean
        
        # Identify non-outliers
        non_outlier_indices = [i for i, dist in enumerate(distances) if dist <= threshold]
        
        # Keep only non-outlier memories
        cluster.memory_list = [cluster.memory_list[i] for i in non_outlier_indices]
        
        # Update center embedding
        if len(cluster.memory_list) > 0:
            cluster.center_embedding = self._calculate_cluster_center(cluster)
    
    def _calculate_cluster_center(self, cluster: Cluster) -> List[float]:
        """
        Calculate the center embedding of a cluster.
        
        Args:
            cluster: Cluster to calculate center for
            
        Returns:
            Center embedding as a list of floats
        """
        memory_embeddings = [np.array(memory.embedding) for memory in cluster.memory_list]
        if not memory_embeddings:
            return []
            
        center = np.mean(memory_embeddings, axis=0).tolist()
        return center
    
    def _clusters_update_strategy(
        self,
        cluster_list: List[Cluster],
        outlier_memory_list: List[Memory],
        new_memory_list: List[Memory],
        cophenetic_distance: float,
        outlier_cutoff_distance: float,
        cluster_merge_distance: float
    ) -> Tuple[List[Cluster], List[Memory]]:
        """
        Update existing clusters with new memories and handle outliers.
        
        Args:
            cluster_list: List of existing Cluster objects
            outlier_memory_list: List of outlier Memory objects from previous run
            new_memory_list: List of new Memory objects to process
            cophenetic_distance: Distance threshold for hierarchical clustering
            outlier_cutoff_distance: Distance threshold to determine outliers
            cluster_merge_distance: Distance threshold for merging clusters
            
        Returns:
            A tuple containing (updated_clusters, new_outlier_memories)
        """
        # Track which clusters were updated
        updated_cluster_ids = set()
        
        # Calculate cluster centers if not already done
        for cluster in cluster_list:
            if not hasattr(cluster, 'cluster_center') or cluster.cluster_center is None:
                cluster.cluster_center = np.array(self._calculate_cluster_center(cluster))
        
        # Assign new memories to nearest clusters or mark as outliers
        for memory in new_memory_list:
            if memory.embedding is None:
                continue
                
            # Find nearest cluster
            nearest_cluster, distance = self._find_nearest_cluster(cluster_list, memory)
            
            # If close enough, add to cluster
            if distance < outlier_cutoff_distance:
                nearest_cluster.memory_list.append(memory)
                updated_cluster_ids.add(nearest_cluster.id)
            else:
                outlier_memory_list.append(memory)
        
        # Merge close clusters
        merge_cluster_ids_list, merge_cluster_list = self._merge_closed_clusters(
            cluster_list, cluster_merge_distance
        )
        
        # Filter out clusters that were merged
        merged_ids = list(itertools.chain(*merge_cluster_ids_list)) if merge_cluster_ids_list else []
        updated_cluster_list = [
            cluster
            for cluster in cluster_list
            if cluster.id in updated_cluster_ids and cluster.id not in merged_ids
        ]
        
        # Determine size threshold for small clusters
        if updated_cluster_list or merge_cluster_list:
            all_clusters = updated_cluster_list + merge_cluster_list
            # size_threshold = math.sqrt(max([len(cluster.memory_list) for cluster in all_clusters]))
            max_size = max([len(cluster.memory_list) for cluster in all_clusters])
            # Use smaller threshold (times 0.8) to keep more clusters
            size_threshold = math.sqrt(max_size) * 0.8
        else:
            # size_threshold = math.sqrt(max([len(cluster.memory_list) for cluster in cluster_list])) if cluster_list else 1
            max_size = max([len(cluster.memory_list) for cluster in cluster_list])
            size_threshold = math.sqrt(max_size) * 0.8
        
        # Process outliers
        if outlier_memory_list:
            outlier_cluster_list, new_outlier_memory_list = self._clusters_initial_strategy(
                outlier_memory_list, cophenetic_distance, size_threshold
            )
        else:
            outlier_cluster_list, new_outlier_memory_list = [], []
        
        # Combine all clusters
        final_cluster_list = updated_cluster_list + merge_cluster_list + outlier_cluster_list
        
        return final_cluster_list, new_outlier_memory_list
    
    def _find_nearest_cluster(self, cluster_list: List[Cluster], memory: Memory) -> Tuple[Cluster, float]:
        """
        Find the nearest cluster to a memory based on embedding distance.
        
        Args:
            cluster_list: List of clusters to search
            memory: Memory to find nearest cluster for
            
        Returns:
            A tuple containing (nearest_cluster, distance_to_cluster)
        """
        if not cluster_list:
            # Should not happen, but handle gracefully
            return None, float('inf')
            
        # Calculate distances to each cluster center
        distances = [
            np.linalg.norm(np.array(memory.embedding) - np.array(self._calculate_cluster_center(cluster)))
            for cluster in cluster_list
        ]
        
        # Find nearest cluster
        nearest_idx = np.argmin(distances)
        return cluster_list[nearest_idx], distances[nearest_idx]
    
    def _merge_closed_clusters(
        self, 
        cluster_list: List[Cluster], 
        cluster_merge_distance: float
    ) -> Tuple[List[List[str]], List[Cluster]]:
        """
        Merge clusters that are close to each other based on the distance threshold.
        
        Args:
            cluster_list: List of clusters to check for merging
            cluster_merge_distance: Threshold distance for merging clusters
            
        Returns:
            A tuple containing (list_of_merged_cluster_ids, list_of_merged_clusters)
        """
        if len(cluster_list) <= 1:
            return [], []
            
        # Use the utility function for finding connected components
        connected_clusters_list = self._find_connected_components(
            cluster_list, cluster_merge_distance
        )
        
        # Keep only components with multiple clusters
        connected_clusters_list = [cc for cc in connected_clusters_list if len(cc) > 1]
        
        if not connected_clusters_list:
            return [], []
            
        # Merge each group of connected clusters
        merge_cluster_ids_list = []
        merge_cluster_list = []
        
        for connected_clusters in connected_clusters_list:
            merge_cluster_ids = [cluster.id for cluster in connected_clusters]
            merge_cluster_ids_list.append(merge_cluster_ids)
            
            # Create merged cluster
            merged_cluster = self._merge_clusters(connected_clusters)
            merge_cluster_list.append(merged_cluster)
        
        return merge_cluster_ids_list, merge_cluster_list
    
    def _merge_clusters(self, connected_clusters: List[Cluster]) -> Cluster:
        """
        Merge a list of connected clusters into a single cluster.
        
        Args:
            connected_clusters: List of clusters to merge
            
        Returns:
            A new merged cluster
        """
        if not connected_clusters:
            return None
            
        # Generate a new ID for the merged cluster
        new_id = f"merged-{connected_clusters[0].id}-{len(connected_clusters)}"
        
        # Create new cluster
        new_cluster = Cluster(
            id=new_id,
            name=connected_clusters[0].name,  # Will be updated later with proper topic generation
            metadata={
                "is_new": True,
                "merge_list": [cluster.id for cluster in connected_clusters]
            }
        )
        
        # Add all memories from all clusters
        for cluster in connected_clusters:
            new_cluster.memory_list.extend(cluster.memory_list)
        
        # Calculate center embedding
        new_cluster.center_embedding = self._calculate_cluster_center(new_cluster)
        
        return new_cluster
    
    def _find_connected_components(
        self, 
        cluster_list: List[Cluster], 
        cluster_merge_distance: float
    ) -> List[List[Cluster]]:
        """
        Finds connected components in a list of clusters based on a distance threshold.
        
        Args:
            cluster_list: List of Cluster objects to analyze
            cluster_merge_distance: Maximum distance for clusters to be considered connected
            
        Returns:
            List of connected components, where each component is a list of clusters
        """
        # Use the utility function for finding connected components
        return find_connected_components(
            cluster_list=cluster_list,
            cluster_merge_distance=cluster_merge_distance,
            get_center_fn=self._calculate_cluster_center
        )
    
    def _cold_start(self, notes_list: List[Note]) -> Dict[str, Any]:
        """
        Perform cold start clustering on a list of notes.
        
        Args:
            notes_list: List of Note objects to process
            
        Returns:
            A dictionary containing cluster data
        """
        embedding_matrix, clean_chunks, all_note_ids = self._build_embedding_chunks(notes_list)
        logger.info(
            f"Built embedding matrix with {len(embedding_matrix)} embeddings from {len(notes_list)} notes"
        )
        
        if len(embedding_matrix) == 0:
            logger.warning("No chunks with embeddings found in the notes list")
            return {}
        
        # Generate topic labels for each chunk
        chunks_with_topics = self._generate_topic_from_chunks(clean_chunks)
        
        # If only one chunk, create a single cluster
        if len(embedding_matrix) <= 1:
            chunk = chunks_with_topics[0]
            return {
                "0": {
                    "indices": [0],
                    "docIds": [chunk.document_id],
                    "contents": [chunk.content],
                    "embedding": [chunk.embedding],
                    "chunkIds": [chunk.id],
                    "tags": chunk.tags if hasattr(chunk, 'tags') else [],
                    "topic": chunk.topic if hasattr(chunk, 'topic') else "Unknown Topic",
                    "topicId": 0,
                    "recTimes": 0
                }
            }
        
        # Perform hierarchical clustering
        clusters = self._hierarchical_clustering(
            embedding_matrix, 
            self.default_cophenetic_distance
        )
        
        # Generate detailed cluster data
        cluster_data = self._gen_cluster_data(clusters, chunks_with_topics)
        
        return cluster_data
    
    def _build_embedding_chunks(self, notes_list: List[Note]) -> Tuple[List[List[float]], List[Chunk], List[str]]:
        """
        Build embedding matrix and clean chunks from a list of notes.
        
        Args:
            notes_list: List of Note objects to process
            
        Returns:
            A tuple containing (embedding_matrix, clean_chunks, all_note_ids)
        """
        all_chunks = [chunk for note in notes_list for chunk in note.chunks]
        all_chunks = [chunk for chunk in all_chunks if chunk.embedding is not None]
        all_note_ids = [note.id for note in notes_list]
        
        clean_chunks = []
        embedding_matrix = []
        
        for chunk in all_chunks:
            if isinstance(chunk.embedding, list) and len(chunk.embedding) > 0:
                clean_chunks.append(chunk)
                embedding_matrix.append(chunk.embedding)
        
        return embedding_matrix, clean_chunks, all_note_ids
    
    def _hierarchical_clustering(
        self, 
        embedding_matrix: List[List[float]], 
        cophenetic_distance: float
    ) -> Dict[str, List[int]]:
        """
        Perform hierarchical clustering on embeddings.
        
        Args:
            embedding_matrix: List of embeddings to cluster
            cophenetic_distance: Distance threshold for creating clusters
            
        Returns:
            A dictionary mapping cluster IDs to lists of point indices
        """
        # Convert to numpy array
        embeddings = np.array(embedding_matrix)
        
        # Calculate pairwise cosine distances
        distances = cdist(embeddings, embeddings, 'cosine')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='average')
        
        # Create flat clusters
        labels = fcluster(linkage_matrix, cophenetic_distance, criterion='distance')
        
        # Map points to clusters
        clusters = {}
        for i, label in enumerate(labels):
            cluster_id = str(label - 1)  # 0-indexed
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        return clusters
    
    def _generate_topic_from_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate topics and keywords for each chunk.
        
        Args:
            chunks: List of chunks to generate topics for
            
        Returns:
            List of chunks with added topic and tags information
        """
        chunks = copy.deepcopy(chunks)
        max_retries = 3
        
        for chunk in chunks:
            for attempt in range(max_retries):
                try:
                    messages = [
                        {"role": "system", "content": SYS_TOPICS},
                        {"role": "user", "content": USR_TOPICS.format(chunk=chunk.content)}
                    ]
                    
                    logger.info(f"Generating topic for chunk: {chunk.id} (Attempt {attempt + 1}/{max_retries})")
                    
                    response = self.llm_service.chat_completion(messages)
                    content = response["choices"][0]["message"]["content"]
                    
                    topic, tags = self._parse_response(content, "topic", "tags")
                    chunk.topic = topic
                    chunk.tags = tags
                    break  # exit retry loop on success
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:  # last attempt failed
                        logger.error(f"All attempts failed for chunk: {traceback.format_exc()}")
                        chunk.topic = "Unknown Topic"  # set default value
                        chunk.tags = ["unclassified"]  # set default value
        
        return chunks
    
    def _gen_cluster_data(self, clusters: Dict[str, List[int]], chunks_with_topics: List[Chunk]) -> Dict[str, Dict[str, Any]]:
        """
        Generate detailed cluster data from cluster indices and chunks.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of point indices
            chunks_with_topics: List of chunks with topic information
            
        Returns:
            A dictionary containing detailed information for each cluster
        """
        cluster_data = {}
        docIds = [chunk.document_id for chunk in chunks_with_topics]
        contents = [chunk.content for chunk in chunks_with_topics]
        embeddings = [chunk.embedding for chunk in chunks_with_topics]
        tags = [getattr(chunk, 'tags', []) for chunk in chunks_with_topics]
        topics = [getattr(chunk, 'topic', "Unknown Topic") for chunk in chunks_with_topics]
        chunkIds = [chunk.id for chunk in chunks_with_topics]
        
        topic_id = 0
        for cid, indices in clusters.items():
            c_tags = [tags[i] for i in indices]
            c_topics = [topics[i] for i in indices]
            
            new_tags, new_topic = self._gen_cluster_topic(c_tags, c_topics)
            
            cluster_data[cid] = {
                "indices": indices,
                "docIds": [docIds[i] for i in indices],
                "contents": [contents[i] for i in indices],
                "embedding": [embeddings[i] for i in indices],
                "chunkIds": [chunkIds[i] for i in indices],
                "tags": new_tags,
                "topic": new_topic,
                "topicId": topic_id,
                "recTimes": 0
            }
            topic_id += 1
            
        return cluster_data
    
    def _gen_cluster_topic(self, c_tags: List[List[str]], c_topics: List[str]) -> Tuple[List[str], str]:
        """
        Generate a combined topic and tags for a cluster.
        
        Args:
            c_tags: List of tags from chunks in the cluster
            c_topics: List of topics from chunks in the cluster
            
        Returns:
            A tuple containing (new_tags, new_topic)
        """
        try:
            messages = [
                {"role": "system", "content": SYS_COMB},
                {"role": "user", "content": USR_COMB.format(topics=c_topics, tags=c_tags)}
            ]
            
            response = self.llm_service.chat_completion(messages)
            content = response["choices"][0]["message"]["content"]
            
            new_topic, new_tags = self._parse_response(content, "topic", "tags")
            return new_tags, new_topic
            
        except Exception as e:
            logger.error(f"Error generating cluster topic: {str(e)}", exc_info=True)
            # Use first topic and tags as fallback
            return c_tags[0] if c_tags else [], c_topics[0] if c_topics else "Unknown Topic"
    
    def _parse_response(self, content: str, key1: str, key2: str) -> Tuple[Any, Any]:
        """
        Parse JSON response to extract specific values.
        
        Args:
            content: JSON string to parse
            key1: First key to extract (typically 'topic')
            key2: Second key to extract (typically 'tags')
            
        Returns:
            A tuple containing the values for the two keys
        """
        try:
            # Try to parse as JSON directly
            data = json.loads(content)
            return data.get(key1, ""), data.get(key2, [])
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON part
            try:
                # Find JSON-like structure in the text
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                    return data.get(key1, ""), data.get(key2, [])
                else:
                    logger.warning(f"Could not find JSON in response: {content}")
                    return "", []
            except Exception as e:
                logger.error(f"Error parsing response: {str(e)}", exc_info=True)
                return "", [] 