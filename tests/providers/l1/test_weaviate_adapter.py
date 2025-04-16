import pytest
from unittest.mock import MagicMock, patch
import weaviate
from uuid import uuid4

from app.providers.l1.weaviate_adapter import WeaviateAdapter, InvalidModelError
from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import L1Shade
from app.models.l1.bio import Bio
from app.providers.l1.weaviate_adapter import BIOS_COLLECTION, TOPICS_COLLECTION, CLUSTERS_COLLECTION, SHADES_COLLECTION

@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client with basic functionality"""
    client = MagicMock(spec=weaviate.Client)
    client.data_object = MagicMock()
    client.data_object.create.return_value = {"id": str(uuid4())}
    return client


@pytest.fixture
def weaviate_adapter(mock_weaviate_client):
    """Return a WeaviateAdapter with mocked client"""
    adapter = WeaviateAdapter(client=mock_weaviate_client)
    return adapter


def test_add_topic(weaviate_adapter, mock_weaviate_client):
    """Test adding a Topic to Weaviate"""
    # Setup test data
    user_id = "user_123"
    mock_topic = MagicMock(spec=Topic)
    mock_topic.id = "topic_123"
    mock_topic.name = "Test Topic"
    mock_topic.summary = "Test summary"
    mock_topic.metadata = None
    mock_topic.embedding = [0.1, 0.2, 0.3]

    # Execute
    result_uuid = weaviate_adapter.add_topic(user_id, mock_topic)

    # Verify
    mock_weaviate_client.data_object.create.assert_called_once()
    call_args = mock_weaviate_client.data_object.create.call_args[0]
    
    # Check that the properties dictionary contains expected values
    properties = call_args[0]
    assert properties["user_id"] == user_id
    assert properties["topic_id"] == mock_topic.id
    assert properties["name"] == mock_topic.name
    assert properties["summary"] == mock_topic.summary
    
    # Check that the class name is correct
    assert call_args[1] == TOPICS_COLLECTION
    
    # Check UUID
    assert result_uuid is not None


def test_add_cluster(weaviate_adapter, mock_weaviate_client):
    """Test adding a Cluster to Weaviate"""
    user_id = "user_123"
    mock_cluster = MagicMock(spec=Cluster)
    mock_cluster.id = "cluster_456"
    mock_cluster.topic_id = "topic_123"
    mock_cluster.name = "Test Cluster"
    mock_cluster.summary = "Cluster summary"
    mock_cluster.metadata = None
    mock_cluster.center_embedding = [0.1, 0.2, 0.3]

    result_uuid = weaviate_adapter.add_cluster(user_id, mock_cluster)

    mock_weaviate_client.data_object.create.assert_called_once()
    call_args = mock_weaviate_client.data_object.create.call_args[0]
    
    properties = call_args[0]
    assert properties["user_id"] == user_id
    assert properties["cluster_id"] == mock_cluster.id
    assert properties["topic_id"] == mock_cluster.topic_id
    assert properties["name"] == mock_cluster.name
    assert properties["summary"] == mock_cluster.summary
    
    assert call_args[1] == CLUSTERS_COLLECTION
    assert result_uuid is not None


def test_add_shade(weaviate_adapter, mock_weaviate_client):
    """Test adding a Shade to Weaviate"""
    user_id = "user_123"
    mock_shade = MagicMock(spec=L1Shade)
    mock_shade.id = "shade_789"
    mock_shade.name = "Test Shade"
    mock_shade.summary = "Shade summary"
    mock_shade.confidence = 0.85
    mock_shade.metadata = None
    mock_shade.embedding = [0.1, 0.2, 0.3]

    result_uuid = weaviate_adapter.add_shade(user_id, mock_shade)

    mock_weaviate_client.data_object.create.assert_called_once()
    call_args = mock_weaviate_client.data_object.create.call_args[0]
    
    properties = call_args[0]
    assert properties["user_id"] == user_id
    assert properties["shade_id"] == mock_shade.id
    assert properties["name"] == mock_shade.name
    assert properties["summary"] == mock_shade.summary
    assert properties["confidence"] == mock_shade.confidence
    
    assert call_args[1] == SHADES_COLLECTION
    assert result_uuid is not None


def test_add_biography(weaviate_adapter, mock_weaviate_client):
    """Test adding a Biography to Weaviate"""
    user_id = "user_123"
    bio_id = "bio_101"
    version = 1
    
    mock_bio = MagicMock(spec=Bio)
    mock_bio.id = "bio_internal_id"
    mock_bio.content_first_view = "First person view"
    mock_bio.summary_first_view = "Summary view"
    mock_bio.content = "Biography content"
    mock_bio.metadata = None

    result_uuid = weaviate_adapter.add_biography(user_id, bio_id, mock_bio, version)

    mock_weaviate_client.data_object.create.assert_called_once()
    call_args = mock_weaviate_client.data_object.create.call_args[0]
    
    properties = call_args[0]
    assert properties["user_id"] == user_id
    assert "content" in properties
    
    assert call_args[1] == BIOS_COLLECTION
    assert result_uuid is not None


def test_validate_model(weaviate_adapter):
    """Test model validation method"""
    # Valid model case
    valid_model = MagicMock(spec=Topic)
    valid_model.id = "valid_id"
    valid_model.to_dict = MagicMock(return_value={})
    
    assert weaviate_adapter._validate_model(valid_model) is True
    
    # Invalid model - missing ID
    invalid_model_no_id = MagicMock(spec=Topic)
    invalid_model_no_id.id = ""  # Empty ID
    invalid_model_no_id.to_dict = MagicMock(return_value={})
    
    with pytest.raises(InvalidModelError, match="Model must have an ID"):
        weaviate_adapter._validate_model(invalid_model_no_id)
    
    # Invalid model - missing to_dict method
    invalid_model_no_to_dict = MagicMock(spec=Topic)
    invalid_model_no_to_dict.id = "some_id"
    # Remove to_dict attribute
    del invalid_model_no_to_dict.to_dict
    
    with pytest.raises(InvalidModelError, match="Model must implement to_dict()"):
        weaviate_adapter._validate_model(invalid_model_no_to_dict)


def test_generate_uuid(weaviate_adapter):
    """Test UUID generation method"""
    entity_type = "topic"
    user_id = "user_123"
    entity_id = "topic_456"
    
    uuid1 = weaviate_adapter._generate_uuid(entity_type, user_id, entity_id)
    uuid2 = weaviate_adapter._generate_uuid(entity_type, user_id, entity_id)
    
    # UUIDs should be deterministic based on input parameters
    assert uuid1 is not None
    assert uuid1 == uuid2
    
    # Different entity should yield different UUID
    different_uuid = weaviate_adapter._generate_uuid(entity_type, user_id, "different_id")
    assert different_uuid != uuid1 