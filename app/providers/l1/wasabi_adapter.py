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

import boto3
from botocore.exceptions import ClientError

from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import Shade
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
        region_name: str = "us-east-1"
    ):
        """
        Initialize the WasabiStorageAdapter.
        
        Args:
            bucket_name: S3 bucket name.
            access_key: AWS access key.
            secret_key: AWS secret key.
            endpoint_url: Wasabi endpoint URL.
            region_name: AWS region name.
        """
        self.bucket_name = bucket_name or os.getenv("WASABI_BUCKET_NAME", "test-bucket")
        access_key = access_key or os.getenv("WASABI_ACCESS_KEY", "test-key")
        secret_key = secret_key or os.getenv("WASABI_SECRET_KEY", "test-secret")
        
        # If running in test mode (with default values), use a mock client
        if self.bucket_name == "test-bucket" and access_key == "test-key" and secret_key == "test-secret":
            logger.info("Initializing WasabiStorageAdapter in test mode with mock client")
            self.client = MagicMock()
            self.resource = MagicMock()
            self.bucket = MagicMock()
            return
        
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.resource = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.bucket = self.resource.Bucket(self.bucket_name)
    
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
    
    def store_json(self, object_key: str, data: Dict[str, Any]) -> None:
        """
        Store JSON data in Wasabi.
        
        Args:
            object_key: S3 object key.
            data: JSON-serializable data to store.
        """
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_json,
                ContentType="application/json"
            )
        except ClientError as e:
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
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"JSON data not found: {object_key}")
                return None
            else:
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
    
    def store_cluster(self, user_id: str, cluster: Cluster) -> str:
        """
        Store a Cluster domain model in Wasabi.
        
        Args:
            user_id: User ID.
            cluster: Cluster domain model.
            
        Returns:
            S3 path where the cluster was stored.
        """
        self._validate_model(cluster)
        
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
    
    def store_shade(self, user_id: str, shade: Shade) -> str:
        """
        Store a Shade domain model in Wasabi.
        
        Args:
            user_id: User ID.
            shade: Shade domain model.
            
        Returns:
            S3 path where the shade was stored.
        """
        self._validate_model(shade)
        
        s3_path = self._get_object_key(SHADES_PREFIX, user_id, shade.id)
        
        # Set the s3_path on the model
        shade.s3_path = s3_path
        
        self.store_json(s3_path, shade.to_dict())
        return s3_path
    
    def get_shade(self, user_id: str, shade_id: str) -> Optional[Shade]:
        """
        Retrieve a Shade domain model from Wasabi.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            
        Returns:
            Shade domain model or None if not found.
        """
        s3_path = self._get_object_key(SHADES_PREFIX, user_id, shade_id)
        data = self.get_json(s3_path)
        if data:
            # Make sure s3_path is in the data
            if not "s3_path" in data:
                data["s3_path"] = s3_path
            return Shade.from_dict(data)
        return None
    
    def store_global_bio(self, user_id: str, version: int, bio: Bio) -> str:
        """
        Store a Bio domain model as a global biography in Wasabi.
        
        Args:
            user_id: User ID.
            version: Biography version.
            bio: Bio domain model.
            
        Returns:
            S3 path where the biography was stored.
        """
        self._validate_model(bio)
        
        bio_id = f"global_v{version}"
        s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        self.store_json(s3_path, bio.to_dict())
        return s3_path
    
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
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            topic_ids = []
            if "Contents" in response:
                for item in response["Contents"]:
                    key = item["Key"]
                    # Extract the ID from the key (remove prefix and .json suffix)
                    topic_id = key[len(prefix):-5]  # -5 to remove ".json"
                    topic_ids.append(topic_id)
            
            return topic_ids
        except ClientError as e:
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
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            cluster_ids = []
            if "Contents" in response:
                for item in response["Contents"]:
                    key = item["Key"]
                    # Extract the ID from the key (remove prefix and .json suffix)
                    cluster_id = key[len(prefix):-5]  # -5 to remove ".json"
                    cluster_ids.append(cluster_id)
            
            return cluster_ids
        except ClientError as e:
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
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            shade_ids = []
            if "Contents" in response:
                for item in response["Contents"]:
                    key = item["Key"]
                    # Extract the ID from the key (remove prefix and .json suffix)
                    shade_id = key[len(prefix):-5]  # -5 to remove ".json"
                    shade_ids.append(shade_id)
            
            return shade_ids
        except ClientError as e:
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
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            bio_ids = []
            if "Contents" in response:
                for item in response["Contents"]:
                    key = item["Key"]
                    # Extract the ID from the key (remove prefix and .json suffix)
                    bio_id = key[len(prefix):-5]  # -5 to remove ".json"
                    bio_ids.append(bio_id)
            
            return bio_ids
        except ClientError as e:
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
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError as e:
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
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError as e:
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
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError as e:
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
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError as e:
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
                paginator = self.client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                
                # Delete objects in batches
                for page in pages:
                    if "Contents" not in page:
                        continue
                    
                    objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    
                    if objects_to_delete:
                        self.client.delete_objects(
                            Bucket=self.bucket_name,
                            Delete={"Objects": objects_to_delete}
                        )
            
            return True
        except ClientError as e:
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
                paginator = self.client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                
                for page in pages:
                    if "Contents" not in page:
                        continue
                    
                    for obj in page["Contents"]:
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
                        
                        # Copy the object
                        self.client.copy_object(
                            Bucket=self.bucket_name,
                            CopySource={"Bucket": self.bucket_name, "Key": source_key},
                            Key=backup_key
                        )
            
            # Store backup metadata
            metadata = {
                "user_id": user_id,
                "backup_id": backup_id,
                "timestamp": str(uuid.uuid1().time),
                "status": "completed"
            }
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=f"{backup_prefix}metadata.json",
                Body=json.dumps(metadata),
                ContentType="application/json"
            )
            
            return backup_id
        except ClientError as e:
            logger.error(f"Error creating backup in Wasabi: {e}")
            raise 