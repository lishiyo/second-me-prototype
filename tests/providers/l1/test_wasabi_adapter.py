import pytest
from unittest.mock import MagicMock, patch
import json
import io
from botocore.exceptions import ClientError

from app.providers.l1.wasabi_adapter import WasabiStorageAdapter, InvalidModelError
from app.providers.blob_store import BlobStore
from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import Shade
from app.models.l1.bio import Bio


@pytest.fixture
def mock_blob_store():
    """Return a mock BlobStore instance."""
    mock = MagicMock(spec=BlobStore)
    return mock


@pytest.fixture
def wasabi_adapter(mock_blob_store):
    """Return a WasabiStorageAdapter with a mocked BlobStore."""
    adapter = WasabiStorageAdapter(
        bucket_name="test-bucket",
        access_key="test-key",
        secret_key="test-secret",
        blob_store=mock_blob_store
    )
    return adapter


def test_init():
    """Test WasabiStorageAdapter initialization."""
    # Test with provided BlobStore
    mock_blob_store = MagicMock(spec=BlobStore)
    adapter = WasabiStorageAdapter(
        bucket_name="test-bucket",
        access_key="test-key",
        secret_key="test-secret",
        blob_store=mock_blob_store
    )
    assert adapter.bucket_name == "test-bucket"
    assert adapter.blob_store == mock_blob_store
    
    # Test auto-creating BlobStore
    # Patch BlobStore specifically where it's imported in wasabi_adapter
    with patch('app.providers.l1.wasabi_adapter.BlobStore') as mock_blob_store_class:
        adapter = WasabiStorageAdapter(
            bucket_name="real-bucket",
            access_key="real-key",
            secret_key="real-secret",
            endpoint_url="real-endpoint",
            region_name="real-region"
        )
        
        # Verify that BlobStore was instantiated with the correct parameters
        mock_blob_store_class.assert_called_once_with(
            access_key="real-key",
            secret_key="real-secret",
            bucket="real-bucket",
            region="real-region",
            endpoint="real-endpoint"
        )
        # Verify the adapter holds the created instance
        assert adapter.blob_store == mock_blob_store_class.return_value


def test_store_json(wasabi_adapter, mock_blob_store):
    """Test storing JSON data."""
    object_key = "l1/test/path.json"
    data = {"test": "data"}
    
    wasabi_adapter.store_json(object_key, data)
    
    mock_blob_store.put_object.assert_called_once()
    call_args = mock_blob_store.put_object.call_args[1]
    
    assert call_args["key"] == object_key
    assert b"test" in call_args["data"]
    assert call_args["metadata"] == {"Content-Type": "application/json"}


def test_get_json(wasabi_adapter, mock_blob_store):
    """Test retrieving JSON data."""
    object_key = "l1/test/path.json"
    
    # Configure the mock response
    test_data = json.dumps({"test": "data"}).encode()
    mock_blob_store.get_object.return_value = test_data
    
    result = wasabi_adapter.get_json(object_key)
    
    mock_blob_store.get_object.assert_called_once_with(key=object_key)
    assert result == {"test": "data"}


def test_get_json_not_found(wasabi_adapter, mock_blob_store):
    """Test retrieving JSON data when not found."""
    object_key = "l1/test/nonexistent.json"
    
    # Configure the mock to raise a "not found" exception
    error = Exception("NoSuchKey: The specified key does not exist")
    mock_blob_store.get_object.side_effect = error
    
    result = wasabi_adapter.get_json(object_key)
    
    mock_blob_store.get_object.assert_called_once()
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


def test_delete_user_data(wasabi_adapter, mock_blob_store):
    """Test deleting all user data."""
    user_id = "test_user"
    
    # Set up mock for list_objects
    mock_blob_store.list_objects.return_value = [
        {"Key": "l1/topics/test_user/topic1.json"},
        {"Key": "l1/topics/test_user/topic2.json"},
    ]
    
    result = wasabi_adapter.delete_user_data(user_id)
    
    assert result is True
    assert mock_blob_store.list_objects.call_count == 4  # One for each prefix
    assert mock_blob_store.delete_object.call_count == 8  # Two objects per prefix, 4 prefixes 