import pytest
from unittest.mock import MagicMock, patch
import json
import io
from botocore.exceptions import ClientError

from app.providers.l1.wasabi_adapter import WasabiStorageAdapter, InvalidModelError
from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import Shade
from app.models.l1.bio import Bio


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
    object_key = "l1/test/path.json"
    data = {"test": "data"}
    
    wasabi_adapter.store_json(object_key, data)
    
    mock_s3_client.put_object.assert_called_once()
    call_args = mock_s3_client.put_object.call_args[1]
    
    assert call_args["Bucket"] == "test-bucket"
    assert call_args["Key"] == object_key
    assert "test" in call_args["Body"]


def test_get_json(wasabi_adapter, mock_s3_client):
    """Test retrieving JSON data."""
    object_key = "l1/test/path.json"
    
    # Configure the mock response
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"test": "data"}).encode()
    mock_s3_client.get_object.return_value = {
        "Body": mock_body
    }
    
    result = wasabi_adapter.get_json(object_key)
    
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=object_key
    )
    
    assert result == {"test": "data"}


def test_get_json_not_found(wasabi_adapter, mock_s3_client):
    """Test retrieving JSON data when not found."""
    object_key = "l1/test/nonexistent.json"
    
    # Configure the mock to raise a proper ClientError
    error_response = {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}}
    mock_s3_client.get_object.side_effect = ClientError(error_response, "GetObject")
    
    result = wasabi_adapter.get_json(object_key)
    
    mock_s3_client.get_object.assert_called_once()
    assert result is None


def test_validate_model(wasabi_adapter):
    """Test model validation."""
    # Valid model
    model = MagicMock()
    model.id = "test_id"
    model.to_dict = MagicMock(return_value={})
    
    assert wasabi_adapter._validate_model(model) is True
    
    # Invalid model - no ID
    invalid_model = MagicMock()
    invalid_model.id = ""
    invalid_model.to_dict = MagicMock(return_value={})
    
    with pytest.raises(InvalidModelError):
        wasabi_adapter._validate_model(invalid_model)
        
    # Invalid model - no to_dict method
    invalid_model2 = MagicMock()
    invalid_model2.id = "test_id"
    del invalid_model2.to_dict
    
    with pytest.raises(InvalidModelError):
        wasabi_adapter._validate_model(invalid_model2)


@pytest.fixture
def mock_topic():
    """Return a mock Topic object."""
    topic = MagicMock(spec=Topic)
    topic.id = "topic_1"
    topic.name = "Test Topic"
    topic.summary = "Test summary"
    topic.to_dict.return_value = {
        "id": "topic_1",
        "name": "Test Topic",
        "summary": "Test summary",
        "document_ids": []
    }
    return topic


@pytest.fixture
def mock_cluster():
    """Return a mock Cluster object."""
    cluster = MagicMock(spec=Cluster)
    cluster.id = "cluster_1"
    cluster.name = "Test Cluster"
    cluster.summary = "Test summary"
    cluster.document_ids = ["doc1", "doc2"]
    cluster.to_dict.return_value = {
        "id": "cluster_1",
        "name": "Test Cluster",
        "summary": "Test summary",
        "document_ids": ["doc1", "doc2"]
    }
    return cluster


@pytest.fixture
def mock_shade():
    """Return a mock Shade object."""
    shade = MagicMock(spec=Shade)
    shade.id = "shade_1"
    shade.name = "Test Shade"
    shade.summary = "Test summary"
    shade.to_dict.return_value = {
        "id": "shade_1",
        "name": "Test Shade",
        "summary": "Test summary",
        "source_clusters": []
    }
    return shade


@pytest.fixture
def mock_bio():
    """Return a mock Bio object."""
    bio = MagicMock(spec=Bio)
    bio.id = "bio_1"
    bio.content_first_view = "Test bio"
    bio.summary_first_view = "Test summary"
    bio.to_dict.return_value = {
        "id": "bio_1",
        "content_first_view": "Test bio",
        "summary_first_view": "Test summary",
        "shades_list": []
    }
    return bio


def test_store_topic(wasabi_adapter, mock_topic):
    """Test storing topic data."""
    user_id = "test_user"
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json, \
         patch.object(wasabi_adapter, '_validate_model') as mock_validate:
        
        s3_path = wasabi_adapter.store_topic(user_id, mock_topic)
        
        # Check path format
        assert s3_path.startswith("l1/topics/")
        assert mock_topic.id in s3_path
        assert s3_path.endswith(".json")
        
        # Check validate_model was called
        mock_validate.assert_called_once_with(mock_topic)
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(s3_path, mock_topic.to_dict())


def test_get_topic(wasabi_adapter, mock_topic):
    """Test retrieving a topic."""
    user_id = "test_user"
    topic_id = "topic_1"
    
    # Patch the get_json method
    with patch.object(wasabi_adapter, 'get_json') as mock_get_json, \
         patch.object(Topic, 'from_dict', return_value=mock_topic) as mock_from_dict:
        
        mock_get_json.return_value = mock_topic.to_dict()
        
        result = wasabi_adapter.get_topic(user_id, topic_id)
        
        # Check get_json was called with the correct path
        s3_path = wasabi_adapter._get_object_key("l1/topics/", user_id, topic_id)
        mock_get_json.assert_called_once_with(s3_path)
        
        # Check the result is what we expect
        assert result == mock_topic


def test_store_cluster(wasabi_adapter, mock_cluster):
    """Test storing cluster data."""
    user_id = "test_user"
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json, \
         patch.object(wasabi_adapter, '_validate_model') as mock_validate:
        
        s3_path = wasabi_adapter.store_cluster(user_id, mock_cluster)
        
        # Check path format
        assert s3_path.startswith("l1/clusters/")
        assert mock_cluster.id in s3_path
        assert s3_path.endswith(".json")
        
        # Check validate_model was called
        mock_validate.assert_called_once_with(mock_cluster)
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(s3_path, mock_cluster.to_dict())


def test_store_shade(wasabi_adapter, mock_shade):
    """Test storing shade data."""
    user_id = "test_user"
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json, \
         patch.object(wasabi_adapter, '_validate_model') as mock_validate:
        
        s3_path = wasabi_adapter.store_shade(user_id, mock_shade)
        
        # Check path format
        assert s3_path.startswith("l1/shades/")
        assert mock_shade.id in s3_path
        assert s3_path.endswith(".json")
        
        # Check validate_model was called
        mock_validate.assert_called_once_with(mock_shade)
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(s3_path, mock_shade.to_dict())


def test_store_global_bio(wasabi_adapter, mock_bio):
    """Test storing global biography data."""
    user_id = "test_user"
    version = 1
    
    # Patch the store_json method
    with patch.object(wasabi_adapter, 'store_json') as mock_store_json, \
         patch.object(wasabi_adapter, '_validate_model') as mock_validate:
        
        s3_path = wasabi_adapter.store_global_bio(user_id, version, mock_bio)
        
        # Check path format
        assert s3_path.startswith("l1/bios/")
        assert "global" in s3_path
        assert str(version) in s3_path
        assert s3_path.endswith(".json")
        
        # Check validate_model was called
        mock_validate.assert_called_once_with(mock_bio)
        
        # Check store_json was called correctly
        mock_store_json.assert_called_once_with(s3_path, mock_bio.to_dict())


def test_delete_user_data(wasabi_adapter, mock_s3_client):
    """Test deleting all user data."""
    user_id = "test_user"
    
    # Set up mock for list_objects_v2 and delete_objects
    paginator = MagicMock()
    mock_s3_client.get_paginator.return_value = paginator
    
    # Mock page with contents
    page1 = {
        "Contents": [
            {"Key": "l1/topics/test_user/topic1.json"},
            {"Key": "l1/topics/test_user/topic2.json"},
        ]
    }
    # Mock page without contents
    page2 = {}
    
    paginator.paginate.return_value = [page1, page2]
    
    result = wasabi_adapter.delete_user_data(user_id)
    
    assert result is True
    assert mock_s3_client.get_paginator.call_count == 4  # One for each prefix
    
    # Should be called 4 times, once for each prefix (topics, clusters, shades, bios)
    assert mock_s3_client.delete_objects.call_count == 4 