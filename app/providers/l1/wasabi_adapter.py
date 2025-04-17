"""
WasabiStorageAdapter for L1 layer.

This module provides an adapter for interacting with Wasabi (S3-compatible)
storage for storing and retrieving L1 domain models.
"""
import logging
import json
import io
import os
from typing import Dict, Any, Optional, Union, List
import uuid
from unittest.mock import MagicMock

from app.providers.blob_store import BlobStore
from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import L1Shade
from app.models.l1.bio import Bio

logger = logging.getLogger(__name__)

# Constants for bucket organization
TOPICS_PREFIX = "l1/topics/"
CLUSTERS_PREFIX = "l1/clusters/"
SHADES_PREFIX = "l1/shades/"
BIOS_PREFIX = "l1/bios/"
VERSIONS_PREFIX = "l1/versions/"

class InvalidModelError(Exception):
    """Raised when a domain model fails validation."""
    pass

class WasabiStorageAdapter:
    """
    Adapter for Wasabi S3-compatible storage operations for L1 data.
    
    Provides methods for storing and retrieving L1 domain models including
    topics, clusters, shades, and biographies.
    """
    
    def __init__(
        self,
        bucket_name: str = None,
        access_key: str = None,
        secret_key: str = None,
        endpoint_url: str = "https://s3.wasabisys.com",
        region_name: str = "us-east-1",
        blob_store: Optional[BlobStore] = None
    ):
        """
        Initialize the WasabiStorageAdapter.
        
        Args:
            bucket_name: S3 bucket name.
            access_key: AWS access key.
            secret_key: AWS secret key.
            endpoint_url: Wasabi endpoint URL.
            region_name: AWS region name.
            blob_store: Optional BlobStore instance to use for S3 operations.
        """
        self.bucket_name = bucket_name or os.getenv("WASABI_BUCKET_NAME", "test-bucket")
        self.access_key = access_key or os.getenv("WASABI_ACCESS_KEY", "test-key")
        self.secret_key = secret_key or os.getenv("WASABI_SECRET_KEY", "test-secret")
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        
        # Use the provided BlobStore or create a new one
        if blob_store is not None:
            self.blob_store = blob_store
        else:
            # If running in test mode (with default values), use a mock store
            if self.bucket_name == "test-bucket" and self.access_key == "test-key" and self.secret_key == "test-secret":
                logger.info("Initializing WasabiStorageAdapter in test mode with mock client")
                self.blob_store = MagicMock()
            else:
                # Create a real BlobStore
                self.blob_store = BlobStore(
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    bucket=self.bucket_name,
                    region=self.region_name,
                    endpoint=self.endpoint_url
                )
    
    def _get_object_key(self, prefix: str, user_id: str, object_id: str) -> str:
        """
        Generate an object key based on prefix, user ID and object ID.
        
        Args:
            prefix: Object type prefix.
            user_id: User ID.
            object_id: Object ID.
            
        Returns:
            S3 object key.
        """
        return f"{prefix}{user_id}/{object_id}.json"
    
    def _validate_model(self, model: Union[Topic, Cluster, L1Shade, Bio]) -> bool:
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
    
    def store_json(self, object_key: str, data: Dict[str, Any]) -> None:
        """
        Store JSON data in Wasabi.
        
        Args:
            object_key: S3 object key.
            data: JSON-serializable data to store.
        """
        try:
            data_json = json.dumps(data)
            self.blob_store.put_object(
                key=object_key,
                data=data_json.encode('utf-8'),
                metadata={"Content-Type": "application/json"}
            )
        except Exception as e:
            logger.error(f"Error storing JSON data in Wasabi: {e}")
            raise
    
    def get_json(self, object_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve JSON data from Wasabi.
        
        Args:
            object_key: S3 object key.
            
        Returns:
            JSON data or None if not found.
        """
        try:
            data_bytes = self.blob_store.get_object(key=object_key)
            return json.loads(data_bytes.decode('utf-8'))
        except Exception as e:
            # Check if it's a "not found" error
            if "NoSuchKey" in str(e):
                logger.warning(f"JSON data not found: {object_key}")
                return None
            logger.error(f"Error retrieving JSON data from Wasabi: {e}")
            raise
    
    # Domain model methods
    
    def store_topic(self, user_id: str, topic: Topic) -> str:
        """
        Store a Topic domain model in Wasabi.
        
        Args:
            user_id: User ID.
            topic: Topic domain model.
            
        Returns:
            S3 path where the topic was stored.
        """
        self._validate_model(topic)
        
        s3_path = self._get_object_key(TOPICS_PREFIX, user_id, topic.id)
        
        # Set the s3_path on the model
        topic.s3_path = s3_path
        
        self.store_json(s3_path, topic.to_dict())
        return s3_path
    
    def get_topic(self, user_id: str, topic_id: str) -> Optional[Topic]:
        """
        Retrieve a Topic domain model from Wasabi.
        
        Args:
            user_id: User ID.
            topic_id: Topic ID.
            
        Returns:
            Topic domain model or None if not found.
        """
        s3_path = self._get_object_key(TOPICS_PREFIX, user_id, topic_id)
        data = self.get_json(s3_path)
        if data:
            # Make sure s3_path is in the data
            if not "s3_path" in data:
                data["s3_path"] = s3_path
            return Topic.from_dict(data)
        return None
    
    def store_cluster(self, user_id: str, cluster: Cluster, version: int = None) -> str:
        """
        Store a Cluster domain model in Wasabi.
        
        Args:
            user_id: User ID.
            cluster: Cluster domain model.
            version: Optional version number.
            
        Returns:
            S3 path where the cluster was stored.
        """
        self._validate_model(cluster)
        
        # If version is provided, include it in the S3 path
        if version is not None:
            s3_path = self._get_object_key(CLUSTERS_PREFIX, user_id, f"{cluster.id}_v{version}")
            # Also add version to the metadata if it doesn't exist
            if cluster.metadata is None:
                cluster.metadata = {}
            if "version" not in cluster.metadata:
                cluster.metadata["version"] = version
        else:
            s3_path = self._get_object_key(CLUSTERS_PREFIX, user_id, cluster.id)
        
        # Set the s3_path on the model
        cluster.s3_path = s3_path
        
        self.store_json(s3_path, cluster.to_dict())
        return s3_path
    
    def get_cluster(self, user_id: str, cluster_id: str) -> Optional[Cluster]:
        """
        Retrieve a Cluster domain model from Wasabi.
        
        Args:
            user_id: User ID.
            cluster_id: Cluster ID.
            
        Returns:
            Cluster domain model or None if not found.
        """
        s3_path = self._get_object_key(CLUSTERS_PREFIX, user_id, cluster_id)
        data = self.get_json(s3_path)
        if data:
            # Make sure s3_path is in the data
            if not "s3_path" in data:
                data["s3_path"] = s3_path
            return Cluster.from_dict(data)
        return None
    
    def store_shade(self, user_id: str, shade: L1Shade) -> str:
        """
        Store a L1Shade domain model in Wasabi.
        
        Args:
            user_id: User ID.
            shade: L1Shade domain model.
            
        Returns:
            S3 path where the shade was stored.
        """
        self._validate_model(shade)
        
        s3_path = self._get_object_key(SHADES_PREFIX, user_id, shade.id)
        
        # Set the s3_path on the model
        shade.s3_path = s3_path
        
        self.store_json(s3_path, shade.to_dict())
        return s3_path
    
    def get_shade(self, user_id: str, shade_id: str) -> Optional[L1Shade]:
        """
        Retrieve a L1Shade domain model from Wasabi.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            
        Returns:
            L1Shade domain model or None if not found.
        """
        s3_path = self._get_object_key(SHADES_PREFIX, user_id, shade_id)
        data = self.get_json(s3_path)
        if data:
            # Make sure s3_path is in the data
            if not "s3_path" in data:
                data["s3_path"] = s3_path
            return L1Shade(**data)
        return None
    
    # def store_global_bio(self, user_id: str, version: int, bio: Bio) -> str:
    #     """
    #     Store a Bio domain model as a global biography in Wasabi.
        
    #     Args:
    #         user_id: User ID.
    #         version: Biography version.
    #         bio: Bio domain model.
            
    #     Returns:
    #         S3 path where the biography was stored.
    #     """
    #     self._validate_model(bio)
        
    #     bio_id = f"global_v{version}"
    #     s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
    #     self.store_json(s3_path, bio.to_dict())
    #     return s3_path
    
    def get_global_bio(self, user_id: str, version: int) -> Optional[Bio]:
        """
        Retrieve a global biography Bio domain model from Wasabi.
        
        Args:
            user_id: User ID.
            version: Biography version.
            
        Returns:
            Bio domain model or None if not found.
        """
        bio_id = f"global_v{version}"
        s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        data = self.get_json(s3_path)
        if data:
            return Bio.from_dict(data)
        return None
    
    def store_status_bio(self, user_id: str, timestamp: str, bio: Bio) -> str:
        """
        Store a Bio domain model as a status biography in Wasabi.
        
        Args:
            user_id: User ID.
            timestamp: Timestamp string.
            bio: Bio domain model.
            
        Returns:
            S3 path where the biography was stored.
        """
        self._validate_model(bio)
        
        bio_id = f"status_{timestamp}"
        s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        self.store_json(s3_path, bio.to_dict())
        return s3_path
    
    def get_status_bio(self, user_id: str, timestamp: str) -> Optional[Bio]:
        """
        Retrieve a status biography Bio domain model from Wasabi.
        
        Args:
            user_id: User ID.
            timestamp: Timestamp string.
            
        Returns:
            Bio domain model or None if not found.
        """
        bio_id = f"status_{timestamp}"
        s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        data = self.get_json(s3_path)
        if data:
            return Bio.from_dict(data)
        return None
    
    def list_user_topics(self, user_id: str) -> List[str]:
        """
        List all topic IDs for a user.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of topic IDs.
        """
        prefix = f"{TOPICS_PREFIX}{user_id}/"
        
        try:
            objects = self.blob_store.list_objects(prefix=prefix)
            
            topic_ids = []
            for obj in objects:
                key = obj["Key"]
                # Extract the ID from the key (remove prefix and .json suffix)
                topic_id = key[len(prefix):-5]  # -5 to remove ".json"
                topic_ids.append(topic_id)
            
            return topic_ids
        except Exception as e:
            logger.error(f"Error listing topics in Wasabi: {e}")
            return []
    
    def list_user_clusters(self, user_id: str) -> List[str]:
        """
        List all cluster IDs for a user.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of cluster IDs.
        """
        prefix = f"{CLUSTERS_PREFIX}{user_id}/"
        
        try:
            objects = self.blob_store.list_objects(prefix=prefix)
            
            cluster_ids = []
            for obj in objects:
                key = obj["Key"]
                # Extract the ID from the key (remove prefix and .json suffix)
                cluster_id = key[len(prefix):-5]  # -5 to remove ".json"
                cluster_ids.append(cluster_id)
            
            return cluster_ids
        except Exception as e:
            logger.error(f"Error listing clusters in Wasabi: {e}")
            return []
    
    def list_user_shades(self, user_id: str) -> List[str]:
        """
        List all shade IDs for a user.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of shade IDs.
        """
        prefix = f"{SHADES_PREFIX}{user_id}/"
        
        try:
            objects = self.blob_store.list_objects(prefix=prefix)
            
            shade_ids = []
            for obj in objects:
                key = obj["Key"]
                # Extract the ID from the key (remove prefix and .json suffix)
                shade_id = key[len(prefix):-5]  # -5 to remove ".json"
                shade_ids.append(shade_id)
            
            return shade_ids
        except Exception as e:
            logger.error(f"Error listing shades in Wasabi: {e}")
            return []
    
    def list_user_biographies(self, user_id: str) -> List[str]:
        """
        List all biography IDs for a user.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of biography IDs.
        """
        prefix = f"{BIOS_PREFIX}{user_id}/"
        
        try:
            objects = self.blob_store.list_objects(prefix=prefix)
            
            bio_ids = []
            for obj in objects:
                key = obj["Key"]
                # Extract the ID from the key (remove prefix and .json suffix)
                bio_id = key[len(prefix):-5]  # -5 to remove ".json"
                bio_ids.append(bio_id)
            
            return bio_ids
        except Exception as e:
            logger.error(f"Error listing biographies in Wasabi: {e}")
            return []
    
    def delete_topic(self, user_id: str, topic_id: str) -> bool:
        """
        Delete a topic from Wasabi.
        
        Args:
            user_id: User ID.
            topic_id: Topic ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        object_key = self._get_object_key(TOPICS_PREFIX, user_id, topic_id)
        
        try:
            self.blob_store.delete_object(key=object_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting topic from Wasabi: {e}")
            return False
    
    def delete_cluster(self, user_id: str, cluster_id: str) -> bool:
        """
        Delete a cluster from Wasabi.
        
        Args:
            user_id: User ID.
            cluster_id: Cluster ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        object_key = self._get_object_key(CLUSTERS_PREFIX, user_id, cluster_id)
        
        try:
            self.blob_store.delete_object(key=object_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cluster from Wasabi: {e}")
            return False
    
    def delete_shade(self, user_id: str, shade_id: str) -> bool:
        """
        Delete a shade from Wasabi.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        object_key = self._get_object_key(SHADES_PREFIX, user_id, shade_id)
        
        try:
            self.blob_store.delete_object(key=object_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting shade from Wasabi: {e}")
            return False
    
    def delete_biography(self, user_id: str, bio_id: str) -> bool:
        """
        Delete a biography from Wasabi.
        
        Args:
            user_id: User ID.
            bio_id: Biography ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        object_key = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        
        try:
            self.blob_store.delete_object(key=object_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting biography from Wasabi: {e}")
            return False
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all L1 data for a specific user.
        
        Args:
            user_id: User ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            # List all objects with the user's prefix
            prefixes = [
                f"{TOPICS_PREFIX}{user_id}/",
                f"{CLUSTERS_PREFIX}{user_id}/",
                f"{SHADES_PREFIX}{user_id}/",
                f"{BIOS_PREFIX}{user_id}/"
            ]
            
            for prefix in prefixes:
                objects = self.blob_store.list_objects(prefix=prefix)
                
                # Delete objects one by one
                for obj in objects:
                    self.blob_store.delete_object(key=obj["Key"])
            
            return True
        except Exception as e:
            logger.error(f"Error deleting user data from Wasabi: {e}")
            return False
    
    def create_backup(self, user_id: str, backup_id: str = None) -> str:
        """
        Create a backup of all user's L1 data.
        
        Args:
            user_id: User ID.
            backup_id: Optional backup ID. If None, a UUID will be generated.
            
        Returns:
            Backup ID.
        """
        if backup_id is None:
            backup_id = str(uuid.uuid4())
        
        backup_prefix = f"backups/{user_id}/{backup_id}/"
        
        try:
            # List all user objects
            prefixes = [
                f"{TOPICS_PREFIX}{user_id}/",
                f"{CLUSTERS_PREFIX}{user_id}/",
                f"{SHADES_PREFIX}{user_id}/",
                f"{BIOS_PREFIX}{user_id}/"
            ]
            
            for prefix in prefixes:
                objects = self.blob_store.list_objects(prefix=prefix)
                
                for obj in objects:
                    source_key = obj["Key"]
                    
                    # Determine backup key (maintain original structure under backup prefix)
                    if source_key.startswith(TOPICS_PREFIX):
                        backup_key = source_key.replace(TOPICS_PREFIX, f"{backup_prefix}{TOPICS_PREFIX}")
                    elif source_key.startswith(CLUSTERS_PREFIX):
                        backup_key = source_key.replace(CLUSTERS_PREFIX, f"{backup_prefix}{CLUSTERS_PREFIX}")
                    elif source_key.startswith(SHADES_PREFIX):
                        backup_key = source_key.replace(SHADES_PREFIX, f"{backup_prefix}{SHADES_PREFIX}")
                    elif source_key.startswith(BIOS_PREFIX):
                        backup_key = source_key.replace(BIOS_PREFIX, f"{backup_prefix}{BIOS_PREFIX}")
                    else:
                        continue
                    
                    # Get the object content and store it at the backup location
                    object_data = self.blob_store.get_object(key=source_key)
                    self.blob_store.put_object(key=backup_key, data=object_data)
            
            # Store backup metadata
            metadata = {
                "user_id": user_id,
                "backup_id": backup_id,
                "timestamp": str(uuid.uuid1().time),
                "status": "completed"
            }
            
            self.blob_store.put_object(
                key=f"{backup_prefix}metadata.json",
                data=json.dumps(metadata).encode('utf-8'),
                metadata={"Content-Type": "application/json"}
            )
            
            return backup_id
        except Exception as e:
            logger.error(f"Error creating backup in Wasabi: {e}")
            raise
    
    def get_document(self, user_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve L0-processed document data from Wasabi storage.
        
        Args:
            user_id: The user ID
            document_id: The document ID
            
        Returns:
            Document data with title, summary, keywords, and raw content if available
        """
        try:
            # Prepare the result document data
            document_data = {
                "id": document_id,
                "user_id": user_id,
                "title": "",
                "insight": {},
                "summary": {},
                "keywords": [],
                "has_raw_content": False,
                "raw_content": ""  # Initialize with empty content
            }
            
            # Get document insight from metadata folder
            insight_path = f"tenant/{user_id}/metadata/{document_id}/insight.json"
            insight_data = self.get_json(insight_path)
            
            if insight_data:
                document_data.update({
                    "title": insight_data.get("title", ""),
                    "insight": insight_data
                })
            
            # Get document summary separately
            summary_path = f"tenant/{user_id}/metadata/{document_id}/summary.json"
            summary_data = self.get_json(summary_path)
            
            if summary_data:
                document_data.update({
                    "summary": summary_data,
                    "keywords": summary_data.get("keywords", [])
                })
            
            # Try to get the raw content (may be large)
            try:
                # List raw files for this document
                raw_objects = self.blob_store.list_objects(prefix=f"tenant/{user_id}/raw/{document_id}_")
                
                if raw_objects and len(raw_objects) > 0:
                    # Get filename from the first matching object
                    raw_path = raw_objects[0].get("Key", "")
                    document_data["raw_s3_path"] = raw_path
                    
                    # Actually fetch the raw content
                    try:
                        logger.debug(f"Fetching raw content from {raw_path}")
                        raw_data = self.blob_store.get_object(key=raw_path)
                        if raw_data:
                            # Try to decode as UTF-8 text
                            raw_content = raw_data.decode('utf-8', errors='replace')
                            document_data["raw_content"] = raw_content
                            document_data["has_raw_content"] = True
                            logger.debug(f"Successfully loaded raw content, length: {len(raw_content)}")
                    except Exception as content_error:
                        logger.warning(f"Error fetching raw content for {raw_path}: {content_error}")
            except Exception as e:
                logger.warning(f"Error listing raw objects for document {document_id}: {e}")
                document_data["has_raw_content"] = False
            
            # Always return document_data with at least the basic structure, even if empty
            return document_data
                
        except Exception as e:
            logger.error(f"Error retrieving document from Wasabi: {e}")
            # Return a minimal document structure instead of None
            return {
                "id": document_id,
                "user_id": user_id,
                "title": "",
                "insight": {},
                "summary": {},
                "keywords": [],
                "has_raw_content": False,
                "raw_content": ""
            }

    def set_data(self, path: str, data: Any) -> bool:
        """
        Store data at a specific path.

        Args:
            path: Path to store data at
            data: Data to store

        Returns:
            True if successful, False otherwise
        """
        try:
            data_str = json.dumps(data)
            response = self.blob_store.put_object(
                key=path,
                data=data_str.encode('utf-8'),
                metadata={"Content-Type": "application/json"}
            )
            return response['ResponseMetadata']['HTTPStatusCode'] == 200
        except Exception as e:
            logger.error(f"Error storing data at {path}: {e}")
            return False

    # New storage methods with version support
    
    def store_biography(self, user_id: str, bio_type: str, bio_data: Dict, version: int) -> bool:
        """
        Store GLOBAL biography data with version information.
        
        Args:
            user_id: The user ID
            bio_type: The biography type (e.g., 'global', 'status')
            bio_data: The full biography data as a dictionary
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure the bio_data has version information
        bio_data = bio_data.copy()  # Make a copy to avoid modifying the original
        bio_data['version'] = version
        bio_data['user_id'] = user_id
        
        # Create a path with version information
        bio_id = f"{bio_type}_v{version}"
        object_key = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        
        try:
            self.store_json(object_key, bio_data)
            return True
        except Exception as e:
            logger.error(f"Error storing biography data: {e}")
            return False
    
    def store_cluster_data(self, user_id: str, cluster_id: str, cluster_data: Dict, version: int) -> bool:
        """
        Store cluster data with version information.
        
        Args:
            user_id: The user ID
            cluster_id: The cluster ID
            cluster_data: The full cluster data as a dictionary
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure the cluster_data has version information
        cluster_data = cluster_data.copy()  # Make a copy to avoid modifying the original
        cluster_data['version'] = version
        cluster_data['user_id'] = user_id
        
        # Create a path with version and ID information
        object_id = f"{cluster_id}_v{version}"
        object_key = self._get_object_key(CLUSTERS_PREFIX, user_id, object_id)
        
        try:
            self.store_json(object_key, cluster_data)
            return True
        except Exception as e:
            logger.error(f"Error storing cluster data: {e}")
            return False
    
    def store_shade_data(self, user_id: str, shade_id: str, shade_data: Dict, version: int) -> bool:
        """
        Store shade data with version information.
        
        Args:
            user_id: The user ID
            shade_id: The shade ID
            shade_data: The full shade data as a dictionary
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure the shade_data has version information
        shade_data = shade_data.copy()  # Make a copy to avoid modifying the original
        shade_data['version'] = version
        shade_data['user_id'] = user_id
        
        # Create a path with version and ID information
        object_id = f"{shade_id}_v{version}"
        object_key = self._get_object_key(SHADES_PREFIX, user_id, object_id)
        
        try:
            self.store_json(object_key, shade_data)
            return True
        except Exception as e:
            logger.error(f"Error storing shade data: {e}")
            return False
    
    def store_chunk_topics(self, user_id: str, chunk_topics: Dict, version: int) -> str:
        """
        Store chunk topics data with version information.
        
        Args:
            user_id: The user ID
            chunk_topics: The chunk topics data as a dictionary
            version: The L1 version number
            
        Returns:
            S3 path where the chunk topics were stored
        """
        # Add version information
        data = {
            'user_id': user_id,
            'version': version,
            'chunk_topics': chunk_topics
        }
        
        # Create a path with version information
        object_id = f"topics_v{version}"
        object_key = f"{TOPICS_PREFIX}{user_id}/{object_id}.json"
        
        try:
            self.store_json(object_key, data)
            return object_key
        except Exception as e:
            logger.error(f"Error storing chunk topics data: {e}")
            raise 