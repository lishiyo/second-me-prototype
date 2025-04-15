"""
Weaviate adapter for L1 layer.

This module provides an adapter for interacting with the Weaviate vector database
for L1 data including topics, clusters, shades, and biographies.
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Set

import weaviate
from weaviate.util import generate_uuid5

from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import Shade
from app.models.l1.bio import Bio
from app.models.l1.note import Note, Chunk
from app.providers.vector_db import VectorDB
from weaviate.collections.classes.filters import Filter
            
# Constants
TOPICS_COLLECTION = "TenantTopic"
CLUSTERS_COLLECTION = "TenantCluster"
SHADES_COLLECTION = "TenantShade"
BIOS_COLLECTION = "TenantBiography"

logger = logging.getLogger(__name__)

class InvalidModelError(Exception):
    """Exception raised when a model is invalid."""
    pass

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
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
    
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
    
    def add_topic(self, user_id: str, topic: Topic) -> str:
        """
        Add a Topic domain model to Weaviate.
        
        Args:
            user_id: User ID.
            topic: Topic domain model.
            
        Returns:
            Weaviate UUID of the created object.
        """
        self._validate_model(topic)
        
        object_uuid = self._generate_uuid("topic", user_id, topic.id)
        
        properties = {
            "user_id": user_id,
            "topic_id": topic.id,
            "name": topic.name,
            "summary": topic.summary or ""
        }
        
        if topic.metadata:
            properties["metadata"] = json.dumps(topic.metadata)
        
        try:
            self.client.data_object.create(
                properties,
                TOPICS_COLLECTION,
                object_uuid,
                vector=topic.embedding or []
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding topic to Weaviate: {e}")
            raise
    
    def add_cluster(self, user_id: str, cluster: Cluster) -> str:
        """
        Add a Cluster domain model to Weaviate.
        
        Args:
            user_id: User ID.
            cluster: Cluster domain model.
            
        Returns:
            Weaviate UUID of the created object.
        """
        self._validate_model(cluster)
        
        object_uuid = self._generate_uuid("cluster", user_id, cluster.id)
        
        properties = {
            "user_id": user_id,
            "cluster_id": cluster.id,
            "topic_id": cluster.topic_id or "",
            "name": cluster.name or "",
            "summary": cluster.summary or ""
        }
        
        if cluster.metadata:
            properties["metadata"] = json.dumps(cluster.metadata)
        
        try:
            self.client.data_object.create(
                properties,
                CLUSTERS_COLLECTION,
                object_uuid,
                vector=cluster.center_embedding or []
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding cluster to Weaviate: {e}")
            raise
    
    def add_shade(self, user_id: str, shade: Shade) -> str:
        """
        Add a Shade domain model to Weaviate.
        
        Args:
            user_id: User ID.
            shade: Shade domain model.
            
        Returns:
            Weaviate UUID of the created object.
        """
        self._validate_model(shade)
        
        object_uuid = self._generate_uuid("shade", user_id, shade.id)
        
        properties = {
            "user_id": user_id,
            "shade_id": shade.id,
            "name": shade.name,
            "summary": shade.summary or "",
            "confidence": shade.confidence
        }
        
        if shade.metadata:
            properties["metadata"] = json.dumps(shade.metadata)
        
        try:
            self.client.data_object.create(
                properties,
                SHADES_COLLECTION,
                object_uuid,
                vector=shade.embedding or []
            )
            return object_uuid
        except Exception as e:
            logger.error(f"Error adding shade to Weaviate: {e}")
            raise
    
    def add_biography(self, user_id: str, bio_id: str, bio: Bio, version: int) -> str:
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
        self._validate_model(bio)
        
        object_uuid = self._generate_uuid("bio", user_id, bio_id)
        
        # Prefer third-person view for vector embedding
        content = bio.content_third_view if hasattr(bio, 'content_third_view') else bio.content
        
        properties = {
            "user_id": user_id,
            "bio_id": bio_id,
            "content": content,
            "version": version
        }
        
        # Ensure we have an embedding
        embedding = getattr(bio, 'embedding', None) or []
        
        if hasattr(bio, 'metadata') and bio.metadata:
            properties["metadata"] = json.dumps(bio.metadata)
        
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
    
    def _validate_model(self, model: Union[Topic, Cluster, Shade, Bio]) -> bool:
        """
        Validate a domain model before storage.
        
        Args:
            model: Domain model to validate.
            
        Returns:
            True if model is valid, raises InvalidModelError otherwise.
        """
        # Basic validation
        if not hasattr(model, 'id') or not model.id:
            raise InvalidModelError("Model must have an ID")
            
        if not hasattr(model, 'to_dict') or not callable(model.to_dict):
            raise InvalidModelError("Model must implement to_dict() method")
            
        return True
    
    def search_topics_models(self, user_id: str, query_embedding: List[float], 
                            limit: int = 10) -> List[Topic]:
        """
        Search for topics by similarity and return domain models.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            limit: Maximum number of results.
            
        Returns:
            List of matching Topic domain models with similarity score added to metadata.
        """
        topic_dicts = self.search_topics(user_id, query_embedding, limit)
        topics = []
        
        for topic_dict in topic_dicts:
            # Create Topic domain model
            topic = Topic(
                id=topic_dict.get("topic_id"),
                name=topic_dict.get("name", ""),
                summary=topic_dict.get("summary", ""),
                embedding=topic_dict.get("embedding", [])
            )
            
            # Add metadata if available
            if "metadata" in topic_dict:
                topic.metadata = topic_dict["metadata"]
                
            # Add certainty score to metadata
            if "certainty" in topic_dict:
                if not topic.metadata:
                    topic.metadata = {}
                topic.metadata["certainty"] = topic_dict["certainty"]
                
            topics.append(topic)
            
        return topics
    
    def search_clusters_models(self, user_id: str, query_embedding: List[float], 
                              topic_id: Optional[str] = None, 
                              limit: int = 10) -> List[Cluster]:
        """
        Search for clusters by similarity and return domain models.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            topic_id: Optional topic ID to filter clusters.
            limit: Maximum number of results.
            
        Returns:
            List of matching Cluster domain models with similarity score added to metadata.
        """
        cluster_dicts = self.search_clusters(user_id, query_embedding, topic_id, limit)
        clusters = []
        
        for cluster_dict in cluster_dicts:
            # Create Cluster domain model
            cluster = Cluster(
                id=cluster_dict.get("cluster_id"),
                topic_id=cluster_dict.get("topic_id"),
                name=cluster_dict.get("name", ""),
                summary=cluster_dict.get("summary", ""),
                center_embedding=cluster_dict.get("embedding", [])
            )
            
            # Add metadata if available
            if "metadata" in cluster_dict:
                cluster.metadata = cluster_dict["metadata"]
                
            # Add certainty score to metadata
            if "certainty" in cluster_dict:
                if not cluster.metadata:
                    cluster.metadata = {}
                cluster.metadata["certainty"] = cluster_dict["certainty"]
                
            clusters.append(cluster)
            
        return clusters
    
    def search_shades_models(self, user_id: str, query_embedding: List[float], 
                            min_confidence: float = 0.0,
                            limit: int = 10) -> List[Shade]:
        """
        Search for shades by similarity and return domain models.
        
        Args:
            user_id: User ID.
            query_embedding: Query vector embedding.
            min_confidence: Minimum confidence score to include in results.
            limit: Maximum number of results.
            
        Returns:
            List of matching Shade domain models with similarity score added to metadata.
        """
        shade_dicts = self.search_shades(user_id, query_embedding, min_confidence, limit)
        shades = []
        
        for shade_dict in shade_dicts:
            # Create Shade domain model
            shade = Shade(
                id=shade_dict.get("shade_id"),
                name=shade_dict.get("name", ""),
                summary=shade_dict.get("summary", ""),
                confidence=shade_dict.get("confidence", 0.0),
                embedding=shade_dict.get("embedding", [])
            )
            
            # Add metadata if available
            if "metadata" in shade_dict:
                shade.metadata = shade_dict["metadata"]
                
            # Add certainty score to metadata
            if "certainty" in shade_dict:
                if not shade.metadata:
                    shade.metadata = {}
                shade.metadata["certainty"] = shade_dict["certainty"]
                
            shades.append(shade)
            
        return shades
    
    def get_similar_shades_models(self, user_id: str, shade_id: str, 
                                 limit: int = 5) -> List[Shade]:
        """
        Get similar shades to a given shade and return domain models.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID to find similar shades for.
            limit: Maximum number of results.
            
        Returns:
            List of similar Shade domain models with similarity score added to metadata.
        """
        similar_dicts = self.get_similar_shades(user_id, shade_id, limit)
        return self._convert_shade_dicts_to_models(similar_dicts)
    
    def _convert_shade_dicts_to_models(self, shade_dicts: List[Dict[str, Any]]) -> List[Shade]:
        """
        Convert a list of shade dictionaries to domain models.
        
        Args:
            shade_dicts: List of shade dictionaries.
            
        Returns:
            List of Shade domain models.
        """
        shades = []
        
        for shade_dict in shade_dicts:
            # Create Shade domain model
            shade = Shade(
                id=shade_dict.get("shade_id"),
                name=shade_dict.get("name", ""),
                summary=shade_dict.get("summary", ""),
                confidence=shade_dict.get("confidence", 0.0),
                embedding=shade_dict.get("embedding", [])
            )
            
            # Add metadata if available
            if "metadata" in shade_dict:
                shade.metadata = shade_dict["metadata"]
                
            # Add similarity score to metadata
            if "similarity" in shade_dict:
                if not shade.metadata:
                    shade.metadata = {}
                shade.metadata["similarity"] = shade_dict["similarity"]
                
            shades.append(shade)
            
        return shades

    def get_document_embedding(self, user_id: str, document_id: str) -> Optional[List[float]]:
        """
        Get the embedding for a document.
        
        Args:
            user_id: The user ID
            document_id: The document ID
            
        Returns:
            Document embedding vector or None if not found
        """
        try:
            collection = self.client.collections.get("Document")
            tenant_collection = collection.with_tenant(user_id)
            
            # Build filter using Filter API
            document_filter = Filter.by_property("document_id").equal(document_id)
            
            # Execute the query
            result = tenant_collection.query.fetch_objects(
                filters=document_filter,
                limit=1,
                include_vector=True
            )
            
            if result.objects and len(result.objects) > 0:
                vector = result.objects[0].vector
                
                # Use the VectorDB utility method to handle dictionary embeddings
                from app.providers.vector_db import VectorDB
                return VectorDB.extract_embedding_from_dict(vector, f"document {document_id}")
            
            self.logger.warning(f"No document embedding found for document {document_id} for user {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting document embedding: {e}")
            return None
            
    def get_document_chunks(self, user_id: str, document_id: str) -> List[Any]:
        """
        Get all chunks for a document with their content and embeddings.
        
        Args:
            user_id: The user ID
            document_id: The document ID
            
        Returns:
            List of chunk objects with content, metadata and embeddings
        """
        try:
            # Get tenant collection
            collection = self.client.collections.get("TenantChunk")
            tenant_collection = collection.with_tenant(user_id)
            
            # Build filter for document chunks using Filter API
            document_filter = Filter.by_property("document_id").equal(document_id)
            
            # Execute the query - include topic and tags in properties
            result = tenant_collection.query.fetch_objects(
                filters=document_filter,
                return_properties=["document_id", "s3_path", "chunk_index", "filename", 
                                  "content_type", "timestamp", "topic", "tags"],
                include_vector=True # this will return a dictionary embedding {"default": [1,2,3]}
            )
            
            if not result.objects:
                self.logger.warning(f"No chunks found for document {document_id} in Weaviate")
                return []
            
            # Import wasabi adapter here to avoid circular imports
            from app.providers.blob_store import BlobStore
            blob_store = BlobStore()
            
            # Convert to objects with expected properties
            chunk_objects = []
            for obj in result.objects:
                props = obj.properties
                chunk_index = props.get("chunk_index", 0)
                # Use consistent ID generation
                chunk_id = VectorDB.generate_consistent_id(user_id, document_id, chunk_index)
                s3_path = props.get("s3_path")
                
                # Get the actual content from Wasabi
                content = ""
                if s3_path:
                    try:
                        content_bytes = blob_store.get_object(s3_path)
                        content = content_bytes.decode('utf-8')
                    except Exception as e:
                        self.logger.error(f"Error loading chunk content from S3 path {s3_path}: {e}")
                        # Continue with empty content if we can't load it
                
                # Process embedding if it's in dictionary format
                embedding = obj.vector
                
                # Use VectorDB util to extract the embedding vector
                embedding = VectorDB.extract_embedding_from_dict(embedding, f"chunk {chunk_id}")

                # Create a chunk object with properties matching what L1 expects
                # Use properties from Weaviate if available, otherwise empty defaults
                chunk_obj = Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    embedding=embedding,
                    tags=props.get("tags", []),  # Use tags from properties if available
                    topic=props.get("topic", ""),  # Use topic from properties if available
                    chunk_index=chunk_index
                )
                chunk_objects.append(chunk_obj)
                
            return chunk_objects
        except Exception as e:
            self.logger.error(f"Error getting document chunks: {e}")
            return []
            
    def get_chunk_embeddings_by_document(self, user_id: str, document_id: str) -> Dict[str, List[float]]:
        """
        Get embeddings for all chunks in a document.
        
        Args:
            user_id: The user ID
            document_id: The document ID
            
        Returns:
            Dictionary mapping chunk_id to embedding vector
        """
        try:
            # Get tenant collection
            collection = self.client.collections.get("TenantChunk")
            tenant_collection = collection.with_tenant(user_id)
            
            # Build filter for document chunks using Filter API
            document_filter = Filter.by_property("document_id").equal(document_id)
            
            # Execute the query
            result = tenant_collection.query.fetch_objects(
                filters=document_filter,
                return_properties=["chunk_index"],
                include_vector=True # this will return a dictionary embedding {"default": [1,2,3]}
            )
            
            # Map chunk_id to embedding
            embeddings = {}
            for obj in result.objects:
                chunk_index = obj.properties.get("chunk_index", 0)
                chunk_id = VectorDB.generate_consistent_id(user_id, document_id, chunk_index)
                if obj.vector:
                    # Use the VectorDB utility to extract the embedding vector
                    embeddings[chunk_id] = VectorDB.extract_embedding_from_dict(obj.vector, f"chunk {chunk_id}")
                else:
                    logger.warning(f"No vector found for chunk {chunk_id}")
                    embeddings[chunk_id] = []
            
            return embeddings
        except Exception as e:
            logger.error(f"Error getting chunk embeddings: {e}")
            return {} 