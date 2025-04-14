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
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist

from app.models.l1.note import Note, Chunk
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# LLM Prompt Templates
SYS_TOPICS = """You are an expert clustering and categorization system. Your task is to generate a concise topic label and a set of relevant tags for the following text content."""

USR_TOPICS = """
Based on the following text, please generate:
1. A concise topic that best represents this content (3-5 words)
2. A list of 3-5 relevant tags (single words or short phrases)

Text content:
{chunk}

Please format your response as a JSON object with the following structure:
{{"topic": "The Topic", "tags": ["tag1", "tag2", "tag3"]}}
"""

SYS_COMB = """You are an expert system for combining and consolidating topics and tags from multiple sources."""

USR_COMB = """
I have multiple topics and their associated tags that need to be combined into a single, coherent topic with consolidated tags.

Topics: {topics}
Tags: {tags}

Please create:
1. A single, unified topic that best represents all of these
2. A consolidated list of 5-7 most relevant tags without duplicates

Please format your response as a JSON object with the following structure:
{{"topic": "The Combined Topic", "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]}}
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
        llm_service: Optional[LLMService] = None,
        default_cophenetic_distance: float = 0.7,
        default_outlier_cutoff_distance: float = 0.9,
        default_cluster_merge_distance: float = 0.75
    ):
        """
        Initialize the TopicsGenerator.
        
        Args:
            llm_service: Service for LLM interactions
            default_cophenetic_distance: Distance threshold for hierarchical clustering
            default_outlier_cutoff_distance: Distance threshold to determine outliers
            default_cluster_merge_distance: Distance threshold for merging clusters
        """
        self.llm_service = llm_service or LLMService()
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
            old_cluster_list: List of existing clusters
            old_outlier_memory_list: List of outlier memories from previous run
            new_memory_list: List of new memories to process
            cophenetic_distance: Distance threshold for hierarchical clustering
            outlier_cutoff_distance: Distance threshold to determine outliers
            cluster_merge_distance: Distance threshold for merging clusters
            
        Returns:
            A dictionary containing updated cluster list and outlier memory list
        """
        # Use default values if not provided
        cophenetic_distance = cophenetic_distance or self.default_cophenetic_distance
        outlier_cutoff_distance = outlier_cutoff_distance or self.default_outlier_cutoff_distance
        cluster_merge_distance = cluster_merge_distance or self.default_cluster_merge_distance
        
        logger.info(f"Generating topics for shades with {len(new_memory_list)} new memories")
        
        # If no existing clusters, perform cold start
        if not old_cluster_list or len(old_cluster_list) == 0:
            logger.info("No existing clusters, performing cold start")
            # Transform memory list into a format suitable for clustering
            notes_list = self._convert_memories_to_notes(new_memory_list)
            cluster_data = self.generate_topics(notes_list)
            
            # Transform the cluster_data into the expected output format
            result = {
                "clusterList": [],
                "outlierMemoryList": []
            }
            
            if cluster_data:
                for cluster_id, cluster in cluster_data.items():
                    cluster_obj = {
                        "clusterId": cluster_id,
                        "topic": cluster["topic"],
                        "tags": cluster["tags"],
                        "memoryList": [new_memory_list[i] for i in cluster["indices"]]
                    }
                    result["clusterList"].append(cluster_obj)
            
            return result
        
        # Incremental update (placeholder for now)
        logger.info("Performing incremental update with existing clusters")
        # TODO: Implement incremental update logic
        
        # Return placeholder result
        return {
            "clusterList": old_cluster_list,
            "outlierMemoryList": old_outlier_memory_list
        }
    
    def _convert_memories_to_notes(self, memory_list: List[Dict[str, Any]]) -> List[Note]:
        """
        Convert memory list to Note objects for processing.
        
        Args:
            memory_list: List of memory dictionaries
            
        Returns:
            List of Note objects
        """
        notes = []
        for memory in memory_list:
            # Create Note from memory
            note = Note(
                id=memory.get("memoryId", ""),
                content=memory.get("content", ""),
                create_time=memory.get("createTime", datetime.now()),
                embedding=memory.get("embedding"),
                title=memory.get("title", ""),
                memory_type=memory.get("memoryType", "TEXT")
            )
            
            # Add chunks if available
            if "chunks" in memory and memory["chunks"]:
                for chunk_data in memory["chunks"]:
                    chunk = Chunk(
                        id=chunk_data.get("id", ""),
                        content=chunk_data.get("content", ""),
                        embedding=chunk_data.get("embedding"),
                        document_id=memory.get("memoryId")
                    )
                    note.chunks.append(chunk)
            
            notes.append(note)
        
        return notes
    
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