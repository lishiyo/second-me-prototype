"""
WasabiStorageAdapter for L1 layer.

This module provides an adapter for interacting with Wasabi (S3-compatible)
storage for storing large blobs of L1 data.
"""
import logging
import json
import io
from typing import Dict, Any, Optional, Union, List
import uuid

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

class WasabiStorageAdapter:
    """
    Adapter for Wasabi S3-compatible storage operations for L1 data.
    
    Provides methods for storing and retrieving large blob data for L1
    including detailed topic data, cluster data, shades, and biographies.
    """
    
    def __init__(
        self,
        bucket_name: str,
        access_key: str,
        secret_key: str,
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
        self.bucket_name = bucket_name
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
        self.bucket = self.resource.Bucket(bucket_name)
    
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
    
    def store_topic_data(self, user_id: str, topic_id: str, data: Dict[str, Any]) -> str:
        """
        Store detailed topic data in Wasabi.
        
        Args:
            user_id: User ID.
            topic_id: Topic ID.
            data: Topic data to store.
            
        Returns:
            S3 object key.
        """
        object_key = self._get_object_key(TOPICS_PREFIX, user_id, topic_id)
        
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_json,
                ContentType="application/json"
            )
            return object_key
        except ClientError as e:
            logger.error(f"Error storing topic data in Wasabi: {e}")
            raise
    
    def get_topic_data(self, user_id: str, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed topic data from Wasabi.
        
        Args:
            user_id: User ID.
            topic_id: Topic ID.
            
        Returns:
            Topic data or None if not found.
        """
        object_key = self._get_object_key(TOPICS_PREFIX, user_id, topic_id)
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Topic data not found: {object_key}")
                return None
            else:
                logger.error(f"Error retrieving topic data from Wasabi: {e}")
                raise
    
    def store_cluster_data(self, user_id: str, cluster_id: str, data: Dict[str, Any]) -> str:
        """
        Store detailed cluster data in Wasabi.
        
        Args:
            user_id: User ID.
            cluster_id: Cluster ID.
            data: Cluster data to store.
            
        Returns:
            S3 object key.
        """
        object_key = self._get_object_key(CLUSTERS_PREFIX, user_id, cluster_id)
        
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_json,
                ContentType="application/json"
            )
            return object_key
        except ClientError as e:
            logger.error(f"Error storing cluster data in Wasabi: {e}")
            raise
    
    def get_cluster_data(self, user_id: str, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed cluster data from Wasabi.
        
        Args:
            user_id: User ID.
            cluster_id: Cluster ID.
            
        Returns:
            Cluster data or None if not found.
        """
        object_key = self._get_object_key(CLUSTERS_PREFIX, user_id, cluster_id)
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Cluster data not found: {object_key}")
                return None
            else:
                logger.error(f"Error retrieving cluster data from Wasabi: {e}")
                raise
    
    def store_shade_data(self, user_id: str, shade_id: str, data: Dict[str, Any]) -> str:
        """
        Store detailed shade data in Wasabi.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            data: Shade data to store.
            
        Returns:
            S3 object key.
        """
        object_key = self._get_object_key(SHADES_PREFIX, user_id, shade_id)
        
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_json,
                ContentType="application/json"
            )
            return object_key
        except ClientError as e:
            logger.error(f"Error storing shade data in Wasabi: {e}")
            raise
    
    def get_shade_data(self, user_id: str, shade_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed shade data from Wasabi.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            
        Returns:
            Shade data or None if not found.
        """
        object_key = self._get_object_key(SHADES_PREFIX, user_id, shade_id)
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Shade data not found: {object_key}")
                return None
            else:
                logger.error(f"Error retrieving shade data from Wasabi: {e}")
                raise
    
    def store_biography_data(self, user_id: str, bio_id: str, data: Dict[str, Any]) -> str:
        """
        Store detailed biography data in Wasabi.
        
        Args:
            user_id: User ID.
            bio_id: Biography ID.
            data: Biography data to store.
            
        Returns:
            S3 object key.
        """
        object_key = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_json,
                ContentType="application/json"
            )
            return object_key
        except ClientError as e:
            logger.error(f"Error storing biography data in Wasabi: {e}")
            raise
    
    def get_biography_data(self, user_id: str, bio_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed biography data from Wasabi.
        
        Args:
            user_id: User ID.
            bio_id: Biography ID.
            
        Returns:
            Biography data or None if not found.
        """
        object_key = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Biography data not found: {object_key}")
                return None
            else:
                logger.error(f"Error retrieving biography data from Wasabi: {e}")
                raise
    
    def store_version_data(self, user_id: str, version_id: str, data: Dict[str, Any]) -> str:
        """
        Store version metadata in Wasabi.
        
        Args:
            user_id: User ID.
            version_id: Version ID.
            data: Version metadata to store.
            
        Returns:
            S3 object key.
        """
        object_key = self._get_object_key(VERSIONS_PREFIX, user_id, version_id)
        
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_json,
                ContentType="application/json"
            )
            return object_key
        except ClientError as e:
            logger.error(f"Error storing version data in Wasabi: {e}")
            raise
    
    def get_version_data(self, user_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve version metadata from Wasabi.
        
        Args:
            user_id: User ID.
            version_id: Version ID.
            
        Returns:
            Version metadata or None if not found.
        """
        object_key = self._get_object_key(VERSIONS_PREFIX, user_id, version_id)
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Version data not found: {object_key}")
                return None
            else:
                logger.error(f"Error retrieving version data from Wasabi: {e}")
                raise
    
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
    
    def list_user_versions(self, user_id: str) -> List[str]:
        """
        List all version IDs for a user.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of version IDs.
        """
        prefix = f"{VERSIONS_PREFIX}{user_id}/"
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            version_ids = []
            if "Contents" in response:
                for item in response["Contents"]:
                    key = item["Key"]
                    # Extract the ID from the key (remove prefix and .json suffix)
                    version_id = key[len(prefix):-5]  # -5 to remove ".json"
                    version_ids.append(version_id)
            
            return version_ids
        except ClientError as e:
            logger.error(f"Error listing versions in Wasabi: {e}")
            return []
    
    def delete_topic_data(self, user_id: str, topic_id: str) -> bool:
        """
        Delete topic data from Wasabi.
        
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
            logger.error(f"Error deleting topic data from Wasabi: {e}")
            return False
    
    def delete_cluster_data(self, user_id: str, cluster_id: str) -> bool:
        """
        Delete cluster data from Wasabi.
        
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
            logger.error(f"Error deleting cluster data from Wasabi: {e}")
            return False
    
    def delete_shade_data(self, user_id: str, shade_id: str) -> bool:
        """
        Delete shade data from Wasabi.
        
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
            logger.error(f"Error deleting shade data from Wasabi: {e}")
            return False
    
    def delete_biography_data(self, user_id: str, bio_id: str) -> bool:
        """
        Delete biography data from Wasabi.
        
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
            logger.error(f"Error deleting biography data from Wasabi: {e}")
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
                f"{BIOS_PREFIX}{user_id}/",
                f"{VERSIONS_PREFIX}{user_id}/"
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
                f"{BIOS_PREFIX}{user_id}/",
                f"{VERSIONS_PREFIX}{user_id}/"
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
                        elif source_key.startswith(VERSIONS_PREFIX):
                            backup_key = source_key.replace(VERSIONS_PREFIX, f"{backup_prefix}{VERSIONS_PREFIX}")
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
    
    def store_json(self, user_id: str, s3_path: str, data: Dict[str, Any]) -> None:
        """
        Store JSON data in Wasabi.
        
        Args:
            user_id: User ID.
            s3_path: S3 path to store the data.
            data: JSON-serializable data to store.
        """
        try:
            data_json = json.dumps(data)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=data_json,
                ContentType="application/json"
            )
        except ClientError as e:
            logger.error(f"Error storing JSON data in Wasabi: {e}")
            raise
    
    def get_json(self, user_id: str, s3_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve JSON data from Wasabi.
        
        Args:
            user_id: User ID.
            s3_path: S3 path to retrieve the data from.
            
        Returns:
            JSON data or None if not found.
        """
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            data_json = response["Body"].read().decode("utf-8")
            return json.loads(data_json)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"JSON data not found: {s3_path}")
                return None
            else:
                logger.error(f"Error retrieving JSON data from Wasabi: {e}")
                raise
    
    # Domain model methods
    
    def store_topic(self, user_id: str, topic_id: str, topic_data: Topic) -> str:
        """
        Store a Topic domain model in Wasabi.
        
        Args:
            user_id: User ID.
            topic_id: Topic ID.
            topic_data: Topic domain model.
            
        Returns:
            S3 path where the topic was stored.
        """
        s3_path = self._get_object_key(TOPICS_PREFIX, user_id, topic_id)
        self.store_json(user_id, s3_path, topic_data.to_dict())
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
        data = self.get_json(user_id, s3_path)
        if data:
            return Topic.from_dict(data)
        return None
    
    def store_cluster(self, user_id: str, cluster_id: str, cluster_data: Cluster) -> str:
        """
        Store a Cluster domain model in Wasabi.
        
        Args:
            user_id: User ID.
            cluster_id: Cluster ID.
            cluster_data: Cluster domain model.
            
        Returns:
            S3 path where the cluster was stored.
        """
        s3_path = self._get_object_key(CLUSTERS_PREFIX, user_id, cluster_id)
        self.store_json(user_id, s3_path, cluster_data.to_dict())
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
        data = self.get_json(user_id, s3_path)
        if data:
            return Cluster.from_dict(data)
        return None
    
    def store_shade(self, user_id: str, shade_id: str, shade_data: Shade) -> str:
        """
        Store a Shade domain model in Wasabi.
        
        Args:
            user_id: User ID.
            shade_id: Shade ID.
            shade_data: Shade domain model.
            
        Returns:
            S3 path where the shade was stored.
        """
        s3_path = self._get_object_key(SHADES_PREFIX, user_id, shade_id)
        self.store_json(user_id, s3_path, shade_data.to_dict())
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
        data = self.get_json(user_id, s3_path)
        if data:
            return Shade.from_dict(data)
        return None
    
    def store_global_bio(self, user_id: str, version: int, bio_data: Bio) -> str:
        """
        Store a Bio domain model as a global biography in Wasabi.
        
        Args:
            user_id: User ID.
            version: Biography version.
            bio_data: Bio domain model.
            
        Returns:
            S3 path where the biography was stored.
        """
        bio_id = f"global_v{version}"
        s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        self.store_json(user_id, s3_path, bio_data.to_dict())
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
        data = self.get_json(user_id, s3_path)
        if data:
            return Bio.from_dict(data)
        return None
    
    def store_status_bio(self, user_id: str, timestamp: str, bio_data: Bio) -> str:
        """
        Store a Bio domain model as a status biography in Wasabi.
        
        Args:
            user_id: User ID.
            timestamp: Timestamp string.
            bio_data: Bio domain model.
            
        Returns:
            S3 path where the biography was stored.
        """
        bio_id = f"status_{timestamp}"
        s3_path = self._get_object_key(BIOS_PREFIX, user_id, bio_id)
        self.store_json(user_id, s3_path, bio_data.to_dict())
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
        data = self.get_json(user_id, s3_path)
        if data:
            return Bio.from_dict(data)
        return None 