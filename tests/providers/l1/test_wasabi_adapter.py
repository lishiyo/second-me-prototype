import pytest
from unittest.mock import MagicMock, patch
import json
import io

from app.providers.l1.wasabi_adapter import WasabiStorageAdapter


@pytest.fixture
def mock_s3_client():
    """Return a mock boto3 S3 client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_s3_resource():
    """Return a mock boto3 S3 resource."""
    resource = MagicMock()
    bucket = MagicMock()
    resource.Bucket.return_value = bucket
    return resource


@pytest.fixture
def wasabi_adapter(mock_s3_client, mock_s3_resource):
    """Return a WasabiStorageAdapter with mocked S3 client and resource."""
    with patch('boto3.client', return_value=mock_s3_client), \
         patch('boto3.resource', return_value=mock_s3_resource):
        adapter = WasabiStorageAdapter(
            bucket_name="test-bucket",
            access_key="test-key",
            secret_key="test-secret"
        )
        adapter.client = mock_s3_client
        adapter.resource = mock_s3_resource
        yield adapter


def test_init():
    """Test WasabiStorageAdapter initialization."""
    with patch('boto3.client'), patch('boto3.resource'):
        adapter = WasabiStorageAdapter(
            bucket_name="test-bucket",
            access_key="test-key",
            secret_key="test-secret"
        )
        assert adapter.bucket_name == "test-bucket"


def test_store_json(wasabi_adapter, mock_s3_client):
    """Test storing JSON data."""
    user_id = "test_user"
    s3_path = "l1/test/path.json"
    data = {"test": "data"}
    
    wasabi_adapter.store_json(user_id, s3_path, data)
    
    mock_s3_client.put_object.assert_called_once()
    call_args = mock_s3_client.put_object.call_args[1]
    
    assert call_args["Bucket"] == "test-bucket"
    assert call_args["Key"] == s3_path
    assert b"test" in call_args["Body"].read()


def test_get_json(wasabi_adapter, mock_s3_client):
    """Test retrieving JSON data."""
    user_id = "test_user"
    s3_path = "l1/test/path.json"
    
    # Configure the mock response
    mock_s3_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps({"test": "data"}).encode())
    }
    
    result = wasabi_adapter.get_json(user_id, s3_path)
    
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=s3_path
    )
    
    assert result == {"test": "data"}


def test_get_json_not_found(wasabi_adapter, mock_s3_client):
    """Test retrieving JSON data when not found."""
    user_id = "test_user"
    s3_path = "l1/test/nonexistent.json"
    
    # Configure the mock to raise an exception
    mock_s3_client.get_object.side_effect = Exception("Test error")
    
    result = wasabi_adapter.get_json(user_id, s3_path)
    
    mock_s3_client.get_object.assert_called_once()
    assert result is None


def test_store_topic(wasabi_adapter):
    """Test storing topic data."""
    user_id = "test_user"
    topic_id = "topic_1"
    topic_data = {"name": "Test Topic", "summary": "Test summary"}
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json:
        s3_path = wasabi_adapter.store_topic(user_id, topic_id, topic_data)
        
        # Check path format
        assert s3_path.startswith("l1/topics/")
        assert topic_id in s3_path
        assert s3_path.endswith(".json")
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(user_id, s3_path, topic_data)


def test_store_cluster(wasabi_adapter):
    """Test storing cluster data."""
    user_id = "test_user"
    cluster_id = "cluster_1"
    cluster_data = {"name": "Test Cluster", "documents": ["doc1", "doc2"]}
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json:
        s3_path = wasabi_adapter.store_cluster(user_id, cluster_id, cluster_data)
        
        # Check path format
        assert s3_path.startswith("l1/clusters/")
        assert cluster_id in s3_path
        assert s3_path.endswith(".json")
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(user_id, s3_path, cluster_data)


def test_store_shade(wasabi_adapter):
    """Test storing shade data."""
    user_id = "test_user"
    shade_id = "shade_1"
    shade_data = {"name": "Test Shade", "summary": "Test summary"}
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json:
        s3_path = wasabi_adapter.store_shade(user_id, shade_id, shade_data)
        
        # Check path format
        assert s3_path.startswith("l1/shades/")
        assert shade_id in s3_path
        assert s3_path.endswith(".json")
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(user_id, s3_path, shade_data)


def test_store_global_bio(wasabi_adapter):
    """Test storing global biography data."""
    user_id = "test_user"
    version = 1
    bio_data = {"content": "Test bio", "summary": "Test summary"}
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json:
        s3_path = wasabi_adapter.store_global_bio(user_id, version, bio_data)
        
        # Check path format
        assert s3_path.startswith("l1/bios/")
        assert "global" in s3_path
        assert str(version) in s3_path
        assert s3_path.endswith(".json")
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(user_id, s3_path, bio_data) 