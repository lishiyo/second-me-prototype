import pytest
from unittest.mock import MagicMock, patch

from app.processors.l1.l1_manager import L1Manager
from app.processors.l1.l1_generator import MergeShadeResult
from app.models.l1.generation_result import L1GenerationResult
from app.models.l1.bio import Bio
from app.models.l1.note import Note
from app.models.l1.shade import Shade as L1Shade


@pytest.fixture
def mock_postgres_adapter():
    """Return a mock PostgreSQL adapter."""
    mock = MagicMock()
    mock.get_latest_version.return_value = 1
    mock.create_version.return_value = MagicMock()
    mock.update_version_status.return_value = None
    return mock


@pytest.fixture
def mock_wasabi_adapter():
    """Return a mock Wasabi adapter."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_weaviate_adapter():
    """Return a mock Weaviate adapter."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_l1_generator():
    """Return a mock L1Generator."""
    mock = MagicMock()
    
    # Set up common return values
    mock.generate_topics.return_value = {"0": {"topic": "Test Topic"}}
    mock.gen_topics_for_shades.return_value = {
        "clusterList": [
            {
                "clusterId": "cluster_0",
                "topic": "Test Topic",
                "memoryList": [
                    {"memoryId": "note1"},
                    {"memoryId": "note2"}
                ]
            }
        ],
        "outlierMemoryList": []
    }
    mock.gen_shade_for_cluster.return_value = L1Shade(
        id="shade_0",
        user_id="test_user",
        name="Test Shade",
        summary="Test summary",
        confidence=0.9,
        s3_path="test/path"
    )
    mock.merge_shades.return_value = MergeShadeResult(
        success=True,
        merge_shade_list=[{
            "id": "merged_shade",
            "name": "Merged Shade",
            "summary": "Merged summary",
            "confidence": 0.9
        }]
    )
    mock.gen_global_biography.return_value = Bio(
        content_third_view="Test global bio",
        summary_third_view="Test summary"
    )
    mock.gen_status_biography.return_value = Bio(
        content_third_view="Test status bio",
        summary_third_view="Test status summary"
    )
    
    return mock


@pytest.fixture
def l1_manager(mock_postgres_adapter, mock_wasabi_adapter, mock_weaviate_adapter, mock_l1_generator):
    """Return an L1Manager with mock dependencies."""
    return L1Manager(
        postgres_adapter=mock_postgres_adapter,
        wasabi_adapter=mock_wasabi_adapter,
        weaviate_adapter=mock_weaviate_adapter,
        l1_generator=mock_l1_generator
    )


@pytest.fixture
def sample_notes():
    """Return sample notes for testing."""
    return [
        Note(
            id="note1", 
            title="Test Note 1", 
            content="This is test note 1", 
            create_time="2023-01-01T00:00:00Z",
            embedding=[0.1, 0.2, 0.3]
        ),
        Note(
            id="note2", 
            title="Test Note 2", 
            content="This is test note 2", 
            create_time="2023-01-02T00:00:00Z",
            embedding=[0.2, 0.3, 0.4]
        )
    ]


def test_init():
    """Test L1Manager initialization."""
    manager = L1Manager()
    assert hasattr(manager, 'postgres_adapter')
    assert hasattr(manager, 'wasabi_adapter')
    assert hasattr(manager, 'weaviate_adapter')
    assert hasattr(manager, 'l1_generator')


@patch('app.processors.l1.l1_manager.L1Manager._store_l1_data')
@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_generate_l1_from_l0_success(mock_extract, mock_store, l1_manager, mock_postgres_adapter, mock_l1_generator, sample_notes):
    """Test successful L1 generation."""
    user_id = "test_user"
    
    # Set up the extract notes mock
    mock_extract.return_value = (sample_notes, [{"memoryId": note.id} for note in sample_notes])
    
    # Test the method
    result = l1_manager.generate_l1_from_l0(user_id)
    
    # Check calls to dependencies
    mock_postgres_adapter.get_latest_version.assert_called_once_with(user_id)
    mock_postgres_adapter.create_version.assert_called_once()
    mock_extract.assert_called_once_with(user_id)
    
    # Check calls to L1Generator
    mock_l1_generator.gen_topics_for_shades.assert_called_once()
    mock_l1_generator.generate_topics.assert_called_once_with(sample_notes)
    mock_l1_generator.gen_shade_for_cluster.assert_called_once()
    mock_l1_generator.merge_shades.assert_called_once()
    mock_l1_generator.gen_global_biography.assert_called_once()
    
    # Check final storage and version completion
    mock_store.assert_called_once()
    mock_postgres_adapter.update_version_status.assert_called_with(
        user_id, 2, "completed"  # Version 2 because get_latest_version returns 1
    )
    
    # Check result
    assert isinstance(result, L1GenerationResult)
    assert result.status == "completed"
    assert result.bio is not None
    assert result.clusters is not None
    assert result.chunk_topics is not None


@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_generate_l1_from_l0_no_notes(mock_extract, l1_manager, mock_postgres_adapter):
    """Test L1 generation with no valid notes."""
    user_id = "test_user"
    
    # Set up the extract notes mock to return empty lists
    mock_extract.return_value = ([], [])
    
    # Test the method
    result = l1_manager.generate_l1_from_l0(user_id)
    
    # Check calls to dependencies
    mock_postgres_adapter.get_latest_version.assert_called_once_with(user_id)
    mock_postgres_adapter.create_version.assert_called_once()
    mock_extract.assert_called_once_with(user_id)
    
    # Check version failure update
    mock_postgres_adapter.update_version_status.assert_called_with(
        user_id, 2, "failed", "No valid documents found for processing"
    )
    
    # Check result
    assert isinstance(result, L1GenerationResult)
    assert result.status == "failed"
    assert result.error == "No valid documents found for processing"


@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_generate_l1_from_l0_exception_in_generation(mock_extract, l1_manager, mock_postgres_adapter, mock_l1_generator, sample_notes):
    """Test handling exception during L1 generation."""
    user_id = "test_user"
    
    # Set up the extract notes mock
    mock_extract.return_value = (sample_notes, [{"memoryId": note.id} for note in sample_notes])
    
    # Make the generator raise an exception
    mock_l1_generator.gen_topics_for_shades.side_effect = Exception("Test error")
    
    # Test the method
    result = l1_manager.generate_l1_from_l0(user_id)
    
    # Check version failure update
    mock_postgres_adapter.update_version_status.assert_called_with(
        user_id, 2, "failed", "Error in L1 generation: Test error"
    )
    
    # Check result
    assert isinstance(result, L1GenerationResult)
    assert result.status == "failed"
    assert "Test error" in result.error


def test_extract_notes_from_l0(l1_manager):
    """Test extracting notes from L0."""
    # This requires a more complex setup - mocking database queries
    # We'll create a basic test for now that just verifies the method exists
    user_id = "test_user"
    notes, memory_list = l1_manager._extract_notes_from_l0(user_id)
    
    # For now, we expect empty results due to our minimal setup
    assert isinstance(notes, list)
    assert isinstance(memory_list, list)
    assert len(notes) == 0
    assert len(memory_list) == 0


def test_store_l1_data(l1_manager, mock_postgres_adapter, mock_wasabi_adapter, mock_weaviate_adapter):
    """Test storing L1 data."""
    # This requires a more complex setup - mocking storage operations
    # Basic test to verify the method exists
    user_id = "test_user"
    version = 1
    bio = Bio(
        content_third_view="Test bio",
        summary_third_view="Test summary"
    )
    clusters = {
        "clusterList": [
            {"clusterId": "cluster_0", "topic": "Test Topic"}
        ]
    }
    chunk_topics = {"0": {"topic": "Test Topic"}}
    
    # Should not raise an exception
    l1_manager._store_l1_data(user_id, version, bio, clusters, chunk_topics)


def test_get_latest_global_bio(l1_manager):
    """Test getting the latest global biography."""
    user_id = "test_user"
    
    # Test the method - our current implementation returns None directly
    result = l1_manager.get_latest_global_bio(user_id)
    
    # For this simple test, we expect None as our implementation is a placeholder
    assert result is None


def test_get_latest_status_bio(l1_manager):
    """Test getting the latest status biography."""
    user_id = "test_user"
    
    # Test the method - our current implementation returns None directly
    result = l1_manager.get_latest_status_bio(user_id)
    
    # For this simple test, we expect None as our implementation is a placeholder
    assert result is None


def test_get_latest_l1_version(l1_manager, mock_postgres_adapter):
    """Test getting the latest L1 version."""
    user_id = "test_user"
    
    # Test the method
    result = l1_manager.get_latest_l1_version(user_id)
    
    # Check the mock was called
    mock_postgres_adapter.get_latest_version.assert_called_once_with(user_id)
    
    # Check result matches what the mock returns
    assert result == 1  # From our mock_postgres_adapter fixture 