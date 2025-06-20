"""
WeaviateAdapter for L1 layer.

This module provides an adapter for interacting with the Weaviate vector database
for L1 data including topics, clusters, and shades.
"""
import logging
import json
from typing import List, Dict, Any, Optional, Tuple

import weaviate
from weaviate.util import generate_uuid5

from app.providers.vector_db import VectorDB
from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import Shade
from app.models.l1.bio import Bio

logger = logging.getLogger(__name__)

# Constants for collection names
TOPICS_COLLECTION = "L1Topics"
CLUSTERS_COLLECTION = "L1Clusters"
SHADES_COLLECTION = "L1Shades"
BIOS_COLLECTION = "L1Biographies"

class WeaviateAdapter:
    """
    Adapter for Weaviate vector database operations for L1 data.
    
    Provides methods for storing and retrieving vector embeddings for L1 data
    including topics, clusters, shades, and biographies.
    """
    
    def __init__(self, client: Optional[weaviate.Client] = None):
        """
        Initialize the WeaviateAdapter.
        
        Args:
            client: Weaviate client. If None, a new client is created.
        """
        if client:
            self.client = client
        else:
            vector_db = VectorDB()
            self.client = vector_db.client
    
    def _generate_uuid(self, entity_type: str, user_id: str, entity_id: str) -> str:
        """
        Generate a deterministic UUID for an entity.
        
        Args:
            entity_type: Type of entity (topic, cluster, shade, bio).
            user_id: User ID.
            entity_id: Entity ID.
            
        Returns:
            A deterministic UUID as string.
        """
        return str(generate_uuid5(f"{entity_type}_{user_id}_{entity_id}"))
    
    def add_topic(self, user_id: str, topic_id: str, name: str, summary: str, 
                 embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a topic to Weaviate.
        
        Args:
            user_id: User ID.
            topic_id: Topic ID.
            name: Topic name.
            summary: Topic summary.
            embedding: Vector embedding of the topic.
            metadata: Optional additional metadata.
            
        Returns:
            Weaviate UUID of the created object.
        """
        object_uuid = self._generate_uuid("topic", user_id, topic_id)
        
        properties = {
            "user_id": user_id,
            "topic_id": topic_id,
            "name": name,
            "summary": summary
        }
        
        if metadata:
            properties["metadata"] = json.dumps(metadata)
        
        try:
            self.client.data_object.create(
                properties,
                TOPICS_COLLECTION,
                object_uuid,
                vector=embedding
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding topic to Weaviate: {e}")
            raise
    
    def add_cluster(self, user_id: str, cluster_id: str, topic_id: str, name: str, 
                   summary: str, embedding: List[float], 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a cluster to Weaviate.
        
        Args:
            user_id: User ID.
            cluster_id: Cluster ID.
            topic_id: Parent topic ID.
            name: Cluster name.
            summary: Cluster summary.
            embedding: Vector embedding of the cluster.
            metadata: Optional additional metadata.
            
        Returns:
            Weaviate UUID of the created object.
        """
        object_uuid = self._generate_uuid("cluster", user_id, cluster_id)
        
        properties = {
            "user_id": user_id,
            "cluster_id": cluster_id,
            "topic_id": topic_id,
            "name": name,
            "summary": summary
        }
        
        if metadata:
            properties["metadata"] = json.dumps(metadata)
        
        try:
            self.client.data_object.create(
                properties,
                CLUSTERS_COLLECTION,
                object_uuid,
                vector=embedding
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding cluster to Weaviate: {e}")
            raise
    
    def add_shade(self, user_id: str, shade_id: str, name: str, summary: str, 
                 embedding: List[float], confidence: float,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a shade to Weaviate.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            name: Shade name.
            summary: Shade summary.
            embedding: Vector embedding of the shade.
            confidence: Confidence score for the shade.
            metadata: Optional additional metadata.
            
        Returns:
            Weaviate UUID of the created object.
        """
        object_uuid = self._generate_uuid("shade", user_id, shade_id)
        
        properties = {
            "user_id": user_id,
            "shade_id": shade_id,
            "name": name,
            "summary": summary,
            "confidence": confidence
        }
        
        if metadata:
            properties["metadata"] = json.dumps(metadata)
        
        try:
            self.client.data_object.create(
                properties,
                SHADES_COLLECTION,
                object_uuid,
                vector=embedding
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding shade to Weaviate: {e}")
            raise
    
    def add_biography(self, user_id: str, bio_id: str, content: str, 
                     embedding: List[float], version: int,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a biography to Weaviate.
        
        Args:
            user_id: User ID.
            bio_id: Biography ID.
            content: Biography content.
            embedding: Vector embedding of the biography.
            version: Version number.
            metadata: Optional additional metadata.
            
        Returns:
            Weaviate UUID of the created object.
        """
        object_uuid = self._generate_uuid("bio", user_id, bio_id)
        
        properties = {
            "user_id": user_id,
            "bio_id": bio_id,
            "content": content,
            "version": version
        }
        
        if metadata:
            properties["metadata"] = json.dumps(metadata)
        
        try:
            self.client.data_object.create(
                properties,
                BIOS_COLLECTION,
                object_uuid,
                vector=embedding
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding biography to Weaviate: {e}")
            raise
    
    def search_topics(self, user_id: str, query_embedding: List[float], 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for topics by similarity.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            limit: Maximum number of results.
            
        Returns:
            List of matching topics with similarity score.
        """
        try:
            result = (
                self.client.query
                .get(TOPICS_COLLECTION, ["user_id", "topic_id", "name", "summary", "metadata"])
                .with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueString": user_id
                })
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": 0.7  # Minimum similarity score
                })
                .with_limit(limit)
                .do()
            )
            
            objects = result.get("data", {}).get("Get", {}).get(TOPICS_COLLECTION, [])
            
            # Process results
            topics = []
            for obj in objects:
                topic = {
                    "topic_id": obj.get("topic_id"),
                    "name": obj.get("name"),
                    "summary": obj.get("summary"),
                    "certainty": obj.get("_additional", {}).get("certainty")
                }
                
                # Parse metadata if available
                metadata = obj.get("metadata")
                if metadata:
                    try:
                        topic["metadata"] = json.loads(metadata)
                    except:
                        topic["metadata"] = {}
                
                topics.append(topic)
            
            return topics
        except Exception as e:
            logger.error(f"Error searching topics in Weaviate: {e}")
            return []
    
    def search_clusters(self, user_id: str, query_embedding: List[float], 
                       topic_id: Optional[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for clusters by similarity.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            topic_id: Optional topic ID to filter clusters.
            limit: Maximum number of results.
            
        Returns:
            List of matching clusters with similarity score.
        """
        try:
            # Build where filter
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["user_id"],
                        "operator": "Equal",
                        "valueString": user_id
                    }
                ]
            }
            
            # Add topic filter if provided
            if topic_id:
                where_filter["operands"].append({
                    "path": ["topic_id"],
                    "operator": "Equal",
                    "valueString": topic_id
                })
            
            result = (
                self.client.query
                .get(CLUSTERS_COLLECTION, ["user_id", "cluster_id", "topic_id", "name", "summary", "metadata"])
                .with_where(where_filter)
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": 0.7  # Minimum similarity score
                })
                .with_limit(limit)
                .do()
            )
            
            objects = result.get("data", {}).get("Get", {}).get(CLUSTERS_COLLECTION, [])
            
            # Process results
            clusters = []
            for obj in objects:
                cluster = {
                    "cluster_id": obj.get("cluster_id"),
                    "topic_id": obj.get("topic_id"),
                    "name": obj.get("name"),
                    "summary": obj.get("summary"),
                    "certainty": obj.get("_additional", {}).get("certainty")
                }
                
                # Parse metadata if available
                metadata = obj.get("metadata")
                if metadata:
                    try:
                        cluster["metadata"] = json.loads(metadata)
                    except:
                        cluster["metadata"] = {}
                
                clusters.append(cluster)
            
            return clusters
        except Exception as e:
            logger.error(f"Error searching clusters in Weaviate: {e}")
            return []
    
    def search_shades(self, user_id: str, query_embedding: List[float], 
                     min_confidence: float = 0.0,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for shades by similarity.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            min_confidence: Minimum confidence score to include in results.
            limit: Maximum number of results.
            
        Returns:
            List of matching shades with similarity score.
        """
        try:
            # Build where filter
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["user_id"],
                        "operator": "Equal",
                        "valueString": user_id
                    },
                    {
                        "path": ["confidence"],
                        "operator": "GreaterThanEqual",
                        "valueNumber": min_confidence
                    }
                ]
            }
            
            result = (
                self.client.query
                .get(SHADES_COLLECTION, ["user_id", "shade_id", "name", "summary", "confidence", "metadata"])
                .with_where(where_filter)
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": 0.7  # Minimum similarity score
                })
                .with_limit(limit)
                .do()
            )
            
            objects = result.get("data", {}).get("Get", {}).get(SHADES_COLLECTION, [])
            
            # Process results
            shades = []
            for obj in objects:
                shade = {
                    "shade_id": obj.get("shade_id"),
                    "name": obj.get("name"),
                    "summary": obj.get("summary"),
                    "confidence": obj.get("confidence"),
                    "certainty": obj.get("_additional", {}).get("certainty")
                }
                
                # Parse metadata if available
                metadata = obj.get("metadata")
                if metadata:
                    try:
                        shade["metadata"] = json.loads(metadata)
                    except:
                        shade["metadata"] = {}
                
                shades.append(shade)
            
            return shades
        except Exception as e:
            logger.error(f"Error searching shades in Weaviate: {e}")
            return []
    
    def search_biographies(self, user_id: str, query_embedding: List[float], 
                          version: Optional[int] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for biographies by similarity.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            version: Optional version number filter.
            limit: Maximum number of results.
            
        Returns:
            List of matching biographies with similarity score.
        """
        try:
            # Build where filter
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["user_id"],
                        "operator": "Equal",
                        "valueString": user_id
                    }
                ]
            }
            
            # Add version filter if provided
            if version is not None:
                where_filter["operands"].append({
                    "path": ["version"],
                    "operator": "Equal",
                    "valueNumber": version
                })
            
            result = (
                self.client.query
                .get(BIOS_COLLECTION, ["user_id", "bio_id", "content", "version", "metadata"])
                .with_where(where_filter)
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": 0.7  # Minimum similarity score
                })
                .with_limit(limit)
                .do()
            )
            
            objects = result.get("data", {}).get("Get", {}).get(BIOS_COLLECTION, [])
            
            # Process results
            bios = []
            for obj in objects:
                bio = {
                    "bio_id": obj.get("bio_id"),
                    "content": obj.get("content"),
                    "version": obj.get("version"),
                    "certainty": obj.get("_additional", {}).get("certainty")
                }
                
                # Parse metadata if available
                metadata = obj.get("metadata")
                if metadata:
                    try:
                        bio["metadata"] = json.loads(metadata)
                    except:
                        bio["metadata"] = {}
                
                bios.append(bio)
            
            return bios
        except Exception as e:
            logger.error(f"Error searching biographies in Weaviate: {e}")
            return []
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all L1 data for a specific user.
        
        Args:
            user_id: User ID to delete data for.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            # Delete topics
            self.client.batch.delete_objects(
                class_name=TOPICS_COLLECTION,
                where={
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueString": user_id
                }
            )
            
            # Delete clusters
            self.client.batch.delete_objects(
                class_name=CLUSTERS_COLLECTION,
                where={
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueString": user_id
                }
            )
            
            # Delete shades
            self.client.batch.delete_objects(
                class_name=SHADES_COLLECTION,
                where={
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueString": user_id
                }
            )
            
            # Delete biographies
            self.client.batch.delete_objects(
                class_name=BIOS_COLLECTION,
                where={
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueString": user_id
                }
            )
            
            return True
        except Exception as e:
            logger.error(f"Error deleting user data from Weaviate: {e}")
            return False
    
    def get_similar_shades(self, user_id: str, shade_id: str, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar shades to a given shade.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID to find similar shades for.
            limit: Maximum number of results.
            
        Returns:
            List of similar shades with similarity score.
        """
        try:
            # First get the reference shade's vector
            shade_uuid = self._generate_uuid("shade", user_id, shade_id)
            reference_obj = self.client.data_object.get_by_id(shade_uuid, with_vector=True)
            
            if not reference_obj:
                logger.error(f"Shade with ID {shade_id} not found")
                return []
            
            reference_vector = reference_obj.get("vector")
            
            # Now search for similar shades
            result = (
                self.client.query
                .get(SHADES_COLLECTION, ["user_id", "shade_id", "name", "summary", "confidence", "metadata"])
                .with_where({
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["user_id"],
                            "operator": "Equal",
                            "valueString": user_id
                        },
                        {
                            "path": ["shade_id"],
                            "operator": "NotEqual",
                            "valueString": shade_id  # Exclude the reference shade
                        }
                    ]
                })
                .with_near_vector({
                    "vector": reference_vector,
                    "certainty": 0.7  # Minimum similarity score
                })
                .with_limit(limit)
                .do()
            )
            
            objects = result.get("data", {}).get("Get", {}).get(SHADES_COLLECTION, [])
            
            # Process results
            shades = []
            for obj in objects:
                shade = {
                    "shade_id": obj.get("shade_id"),
                    "name": obj.get("name"),
                    "summary": obj.get("summary"),
                    "confidence": obj.get("confidence"),
                    "similarity": obj.get("_additional", {}).get("certainty")
                }
                
                # Parse metadata if available
                metadata = obj.get("metadata")
                if metadata:
                    try:
                        shade["metadata"] = json.loads(metadata)
                    except:
                        shade["metadata"] = {}
                
                shades.append(shade)
            
            return shades
        except Exception as e:
            logger.error(f"Error getting similar shades from Weaviate: {e}")
            return []
    
    def apply_schema(self) -> bool:
        """
        Apply the Weaviate schema for L1 data.
        
        Returns:
            True if schema was applied successfully, False otherwise.
        """
        try:
            # Define the schema for L1 data
            schema = {
                "classes": [
                    {
                        "class": TOPICS_COLLECTION,
                        "description": "L1 Topics",
                        "vectorizer": "none",  # Use custom vectors
                        "properties": [
                            {
                                "name": "user_id",
                                "dataType": ["string"],
                                "description": "User ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "topic_id",
                                "dataType": ["string"],
                                "description": "Topic ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "name",
                                "dataType": ["string"],
                                "description": "Topic name",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "summary",
                                "dataType": ["text"],
                                "description": "Topic summary",
                                "indexFilterable": False,
                                "indexSearchable": True
                            },
                            {
                                "name": "metadata",
                                "dataType": ["string"],
                                "description": "JSON-encoded metadata",
                                "indexFilterable": False,
                                "indexSearchable": False
                            }
                        ]
                    },
                    {
                        "class": CLUSTERS_COLLECTION,
                        "description": "L1 Clusters",
                        "vectorizer": "none",  # Use custom vectors
                        "properties": [
                            {
                                "name": "user_id",
                                "dataType": ["string"],
                                "description": "User ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "cluster_id",
                                "dataType": ["string"],
                                "description": "Cluster ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "topic_id",
                                "dataType": ["string"],
                                "description": "Parent topic ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "name",
                                "dataType": ["string"],
                                "description": "Cluster name",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "summary",
                                "dataType": ["text"],
                                "description": "Cluster summary",
                                "indexFilterable": False,
                                "indexSearchable": True
                            },
                            {
                                "name": "metadata",
                                "dataType": ["string"],
                                "description": "JSON-encoded metadata",
                                "indexFilterable": False,
                                "indexSearchable": False
                            }
                        ]
                    },
                    {
                        "class": SHADES_COLLECTION,
                        "description": "L1 Shades",
                        "vectorizer": "none",  # Use custom vectors
                        "properties": [
                            {
                                "name": "user_id",
                                "dataType": ["string"],
                                "description": "User ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "shade_id",
                                "dataType": ["string"],
                                "description": "Shade ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "name",
                                "dataType": ["string"],
                                "description": "Shade name",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "summary",
                                "dataType": ["text"],
                                "description": "Shade summary",
                                "indexFilterable": False,
                                "indexSearchable": True
                            },
                            {
                                "name": "confidence",
                                "dataType": ["number"],
                                "description": "Confidence score",
                                "indexFilterable": True,
                                "indexSearchable": False
                            },
                            {
                                "name": "metadata",
                                "dataType": ["string"],
                                "description": "JSON-encoded metadata",
                                "indexFilterable": False,
                                "indexSearchable": False
                            }
                        ]
                    },
                    {
                        "class": BIOS_COLLECTION,
                        "description": "L1 Biographies",
                        "vectorizer": "none",  # Use custom vectors
                        "properties": [
                            {
                                "name": "user_id",
                                "dataType": ["string"],
                                "description": "User ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "bio_id",
                                "dataType": ["string"],
                                "description": "Biography ID",
                                "indexFilterable": True,
                                "indexSearchable": True
                            },
                            {
                                "name": "content",
                                "dataType": ["text"],
                                "description": "Biography content",
                                "indexFilterable": False,
                                "indexSearchable": True
                            },
                            {
                                "name": "version",
                                "dataType": ["number"],
                                "description": "Version number",
                                "indexFilterable": True,
                                "indexSearchable": False
                            },
                            {
                                "name": "metadata",
                                "dataType": ["string"],
                                "description": "JSON-encoded metadata",
                                "indexFilterable": False,
                                "indexSearchable": False
                            }
                        ]
                    }
                ]
            }
            
            # Check if collections already exist
            existing_schema = self.client.schema.get()
            existing_collections = [c["class"] for c in existing_schema["classes"]]
            
            for class_obj in schema["classes"]:
                class_name = class_obj["class"]
                
                if class_name in existing_collections:
                    logger.info(f"Collection {class_name} already exists, skipping")
                    continue
                
                # Create the class
                self.client.schema.create_class(class_obj)
                logger.info(f"Created collection {class_name}")
            
            return True
        except Exception as e:
            logger.error(f"Error applying Weaviate schema: {e}")
            return False
            
    def get_schema_collections(self) -> List[str]:
        """
        Get the list of L1 collections in the schema.
        
        Returns:
            List of collection names.
        """
        return [TOPICS_COLLECTION, CLUSTERS_COLLECTION, SHADES_COLLECTION, BIOS_COLLECTION]

    # Domain model methods
    
    def add_topic_model(self, user_id: str, topic: Topic) -> str:
        """
        Add a Topic domain model to Weaviate.
        
        Args:
            user_id: User ID.
            topic: Topic domain model.
            
        Returns:
            Weaviate UUID of the created object.
        """
        return self.add_topic(
            user_id=user_id,
            topic_id=topic.id,
            name=topic.name,
            summary=topic.summary or "",
            embedding=topic.embedding or [],
            metadata=topic.metadata
        )
    
    def add_cluster_model(self, user_id: str, cluster: Cluster) -> str:
        """
        Add a Cluster domain model to Weaviate.
        
        Args:
            user_id: User ID.
            cluster: Cluster domain model.
            
        Returns:
            Weaviate UUID of the created object.
        """
        return self.add_cluster(
            user_id=user_id,
            cluster_id=cluster.id,
            topic_id=cluster.topic_id or "",
            name=cluster.name or "",
            summary=cluster.summary or "",
            embedding=cluster.center_embedding or [],
            metadata=cluster.metadata
        )
    
    def add_shade_model(self, user_id: str, shade: Shade) -> str:
        """
        Add a Shade domain model to Weaviate.
        
        Args:
            user_id: User ID.
            shade: Shade domain model.
            
        Returns:
            Weaviate UUID of the created object.
        """
        return self.add_shade(
            user_id=user_id,
            shade_id=shade.id,
            name=shade.name,
            summary=shade.summary or "",
            embedding=shade.embedding or [],
            confidence=shade.confidence,
            metadata=shade.metadata
        )
    
    def add_biography_model(self, user_id: str, bio_id: str, bio: Bio, version: int) -> str:
        """
        Add a Bio domain model to Weaviate.
        
        Args:
            user_id: User ID.
            bio_id: Biography ID.
            bio: Bio domain model.
            version: Version number.
            
        Returns:
            Weaviate UUID of the created object.
        """
        # Prefer third-person view for vector embedding
        content = bio.content_third_view if hasattr(bio, 'content_third_view') else bio.content
        
        # Ensure we have an embedding
        embedding = getattr(bio, 'embedding', None) or []
        
        return self.add_biography(
            user_id=user_id,
            bio_id=bio_id,
            content=content,
            embedding=embedding,
            version=version,
            metadata=getattr(bio, 'metadata', {})
        ) 