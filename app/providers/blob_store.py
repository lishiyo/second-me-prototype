import os
import boto3
from botocore.exceptions import ClientError
from typing import BinaryIO, Optional, Dict, List, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class BlobStore:
    """
    A class for interacting with Wasabi S3 storage.
    This adapter provides methods to upload, download, and manage files in Wasabi.
    """
    
    def __init__(self, 
                 access_key: Optional[str] = None, 
                 secret_key: Optional[str] = None, 
                 bucket: Optional[str] = None,
                 region: Optional[str] = None,
                 endpoint: Optional[str] = None):
        """
        Initialize the Wasabi S3 client.
        
        Args:
            access_key: Wasabi access key (defaults to env var WASABI_ACCESS_KEY)
            secret_key: Wasabi secret key (defaults to env var WASABI_SECRET_KEY)
            bucket: Wasabi bucket name (defaults to env var WASABI_BUCKET)
            region: Wasabi region (defaults to env var WASABI_REGION)
            endpoint: Wasabi endpoint URL (defaults to env var WASABI_ENDPOINT)
        """
        self.access_key = access_key or os.environ.get('WASABI_ACCESS_KEY')
        self.secret_key = secret_key or os.environ.get('WASABI_SECRET_KEY')
        self.bucket = bucket or os.environ.get('WASABI_BUCKET')
        self.region = region or os.environ.get('WASABI_REGION', 'us-west-1')
        self.endpoint = endpoint or os.environ.get('WASABI_ENDPOINT')
        
        if not all([self.access_key, self.secret_key, self.bucket, self.endpoint]):
            raise ValueError("Missing required Wasabi configuration")
        
        self.client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            endpoint_url=self.endpoint
        )
        
        self.resource = boto3.resource(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            endpoint_url=self.endpoint
        )
        
        # Ensure the bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """
        Ensure the configured bucket exists, create it if it doesn't.
        """
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                logger.info(f"Creating bucket {self.bucket}")
                self.client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            else:
                raise
    
    def put_object(self, key: str, data: bytes, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Upload a binary object to Wasabi S3.
        
        Args:
            key: S3 key where the object will be stored
            data: Binary data to upload
            metadata: Optional metadata to associate with the object
            
        Returns:
            S3 URI of the uploaded object
        """
        try:
            params = {
                'Bucket': self.bucket,
                'Key': key,
                'Body': data
            }
            if metadata:
                params['Metadata'] = metadata
                
            self.client.put_object(**params)
            return f"s3://{self.bucket}/{key}"
        except ClientError as e:
            logger.error(f"Error uploading object {key}: {e}")
            raise
    
    def put_file(self, key: str, file_path: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Upload a local file to Wasabi S3.
        
        Args:
            key: S3 key where the file will be stored
            file_path: Local path to the file to upload
            metadata: Optional metadata to associate with the object
            
        Returns:
            S3 URI of the uploaded file
        """
        try:
            params = {
                'Bucket': self.bucket,
                'Key': key
            }
            if metadata:
                params['Metadata'] = metadata
                
            self.resource.meta.client.upload_file(file_path, self.bucket, key, ExtraArgs=params)
            return f"s3://{self.bucket}/{key}"
        except ClientError as e:
            logger.error(f"Error uploading file {file_path} to {key}: {e}")
            raise
    
    def put_fileobj(self, key: str, fileobj: BinaryIO, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Upload a file-like object to Wasabi S3.
        
        Args:
            key: S3 key where the file will be stored
            fileobj: File-like object to upload
            metadata: Optional metadata to associate with the object
            
        Returns:
            S3 URI of the uploaded file
        """
        try:
            # Create extra args for upload
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Upload the file - note that boto3 will close the fileobj after upload
            self.client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args)
            
            return f"s3://{self.bucket}/{key}"
        except ClientError as e:
            logger.error(f"Error uploading fileobj to {key}: {e}")
            raise
    
    def get_object(self, key: str) -> bytes:
        """
        Download an object from Wasabi S3.
        
        Args:
            key: S3 key to download
            
        Returns:
            Binary data of the downloaded object
        """
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error downloading object {key}: {e}")
            raise
    
    def get_fileobj(self, key: str, fileobj: BinaryIO) -> None:
        """
        Download an object from Wasabi S3 into a file-like object.
        
        Args:
            key: S3 key to download
            fileobj: File-like object to write the data to
        """
        try:
            self.client.download_fileobj(self.bucket, key, fileobj)
        except ClientError as e:
            logger.error(f"Error downloading object {key} to fileobj: {e}")
            raise
    
    def get_metadata(self, key: str) -> Dict[str, str]:
        """
        Get metadata for an object in Wasabi S3.
        
        Args:
            key: S3 key to get metadata for
            
        Returns:
            Dictionary of metadata
        """
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return response.get('Metadata', {})
        except ClientError as e:
            logger.error(f"Error getting metadata for {key}: {e}")
            raise
    
    def delete_object(self, key: str) -> None:
        """
        Delete an object from Wasabi S3.
        
        Args:
            key: S3 key to delete
        """
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            logger.error(f"Error deleting object {key}: {e}")
            raise
    
    def list_objects(self, prefix: str, delimiter: str = "/") -> List[Dict[str, Any]]:
        """
        List objects in Wasabi S3 with the given prefix.
        
        Args:
            prefix: Prefix to filter objects by
            delimiter: Delimiter character for hierarchical listing
            
        Returns:
            List of object information dictionaries
        """
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            objects = []
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter=delimiter):
                if 'Contents' in page:
                    objects.extend(page['Contents'])
                    
            return objects
        except ClientError as e:
            logger.error(f"Error listing objects with prefix {prefix}: {e}")
            raise
    
    def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate a pre-signed URL for downloading an object.
        
        Args:
            key: S3 key to generate URL for
            expires_in: Expiration time in seconds
            
        Returns:
            Pre-signed URL for downloading the object
        """
        try:
            return self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expires_in
            )
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {key}: {e}")
            raise
    
    def get_upload_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate a pre-signed URL for uploading an object.
        
        Args:
            key: S3 key to generate URL for
            expires_in: Expiration time in seconds
            
        Returns:
            Pre-signed URL for uploading to the object
        """
        try:
            return self.client.generate_presigned_url(
                'put_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expires_in
            )
        except ClientError as e:
            logger.error(f"Error generating presigned upload URL for {key}: {e}")
            raise
    
    @staticmethod
    def parse_s3_uri(uri: str) -> tuple:
        """
        Parse an S3 URI into bucket and key components.
        
        Args:
            uri: S3 URI to parse (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
        """
        parsed = urlparse(uri)
        if parsed.scheme != 's3':
            raise ValueError(f"Not an S3 URI: {uri}")
            
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def uri_exists(self, uri: str) -> bool:
        """
        Check if an S3 URI exists.
        
        Args:
            uri: S3 URI to check
            
        Returns:
            True if the object exists, False otherwise
        """
        try:
            bucket, key = self.parse_s3_uri(uri)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise 