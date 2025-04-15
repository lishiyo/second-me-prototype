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
def mock_topics_generator():
    """Return a mock TopicsGenerator."""
    mock = MagicMock()
    mock.generate_topics.return_value = {"0": {"topic": "Test Topic"}}
    mock.generate_topics_for_shades.return_value = {
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
    return mock


@pytest.fixture
def mock_shade_generator():
    """Return a mock ShadeGenerator."""
    mock = MagicMock()
    mock.generate_shade.return_value = L1Shade(
        id="shade_0",
        user_id="test_user",
        name="Test Shade",
        summary="Test summary",
        confidence=0.9,
        s3_path="test/path"
    )
    return mock


@pytest.fixture
def mock_shade_merger():
    """Return a mock ShadeMerger."""
    mock = MagicMock()
    mock.merge_shades.return_value = MergeShadeResult(
        success=True,
        merge_shade_list=[{
            "id": "merged_shade",
            "name": "Merged Shade",
            "summary": "Merged summary",
            "confidence": 0.9
        }]
    )
    return mock


@pytest.fixture
def mock_biography_generator():
    """Return a mock BiographyGenerator."""
    mock = MagicMock()
    mock.generate_global_biography.return_value = Bio(
        content_third_view="Test global bio",
        summary_third_view="Test summary"
    )
    mock.generate_status_biography.return_value = Bio(
        content_third_view="Test status bio",
        summary_third_view="Test status summary"
    )
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
def l1_manager(
    mock_postgres_adapter, 
    mock_wasabi_adapter, 
    mock_weaviate_adapter, 
    mock_l1_generator,
    mock_topics_generator,
    mock_shade_generator,
    mock_shade_merger,
    mock_biography_generator
):
    """Return an L1Manager with mock dependencies."""
    return L1Manager(
        postgres_adapter=mock_postgres_adapter,
        wasabi_adapter=mock_wasabi_adapter,
        weaviate_adapter=mock_weaviate_adapter,
        l1_generator=mock_l1_generator,
        topics_generator=mock_topics_generator,
        shade_generator=mock_shade_generator,
        shade_merger=mock_shade_merger,
        biography_generator=mock_biography_generator
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
    """Test L1Manager initialization with default dependencies."""
    # Patch dependencies where they are imported by L1Manager
    with patch('app.processors.l1.l1_manager.PostgresAdapter') as mock_pg, \
         patch('app.processors.l1.l1_manager.WasabiStorageAdapter') as mock_wasabi, \
         patch('app.processors.l1.l1_manager.WeaviateAdapter') as mock_weaviate, \
         patch('app.providers.vector_db.VectorDB') as mock_vector_db, \
         patch('app.processors.l1.l1_manager.L1Generator') as mock_gen, \
         patch('app.processors.l1.l1_manager.TopicsGenerator') as mock_topics, \
         patch('app.processors.l1.l1_manager.ShadeGenerator') as mock_shade, \
         patch('app.processors.l1.l1_manager.ShadeMerger') as mock_merger, \
         patch('app.processors.l1.l1_manager.BiographyGenerator') as mock_bio:

        # Instantiate the manager - should call the mock classes
        manager = L1Manager()

        # Assert that the mock classes were called (instantiated)
        mock_pg.assert_called_once()
        mock_wasabi.assert_called_once()
        mock_weaviate.assert_called_once()
        mock_gen.assert_called_once()
        mock_topics.assert_called_once()
        mock_shade.assert_called_once()
        mock_merger.assert_called_once()
        mock_bio.assert_called_once()

        # Assert that the manager has the instances from the mocks
        assert manager.postgres_adapter == mock_pg.return_value
        assert manager.wasabi_adapter == mock_wasabi.return_value
        assert manager.weaviate_adapter == mock_weaviate.return_value
        assert manager.l1_generator == mock_gen.return_value
        assert manager.topics_generator == mock_topics.return_value
        assert manager.shade_generator == mock_shade.return_value
        assert manager.shade_merger == mock_merger.return_value
        assert manager.biography_generator == mock_bio.return_value

        # Keep original assertions too
        assert hasattr(manager, 'postgres_adapter')
        assert hasattr(manager, 'wasabi_adapter')
        assert hasattr(manager, 'weaviate_adapter')
        assert hasattr(manager, 'l1_generator')
        assert hasattr(manager, 'topics_generator')
        assert hasattr(manager, 'shade_generator')
        assert hasattr(manager, 'shade_merger')
        assert hasattr(manager, 'biography_generator')


@patch('app.processors.l1.l1_manager.L1Manager._store_l1_data')
@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_generate_l1_from_l0_success(
    mock_extract, 
    mock_store, 
    l1_manager, 
    mock_postgres_adapter, 
    mock_topics_generator,
    mock_shade_generator,
    mock_shade_merger,
    mock_biography_generator, 
    sample_notes
):
    """Test successful L1 generation."""
    user_id = "test_user"
    
    # Set up the extract notes mock
    mock_extract.return_value = (sample_notes, [{"memoryId": note.id} for note in sample_notes])
    
    # Make sure store returns a proper value to handle the flow of the function
    mock_store.return_value = None
    
    # Test the method
    result = l1_manager.generate_l1_from_l0(user_id)
    
    # Check calls to dependencies
    mock_extract.assert_called_once_with(user_id)
    
    # Check calls to generators
    mock_topics_generator.generate_topics_for_shades.assert_called_once_with(
        user_id=user_id,
        old_cluster_list=[],
        old_outlier_memory_list=[],
        new_memory_list=[{"memoryId": note.id} for note in sample_notes]
    )
    mock_topics_generator.generate_topics.assert_called_once()
    mock_shade_generator.generate_shade.assert_called()
    mock_shade_merger.merge_shades.assert_called_once()
    mock_biography_generator.generate_global_biography.assert_called_once()
    
    # Check that store was called
    mock_store.assert_called_once()
    
    # Check result
    assert isinstance(result, L1GenerationResult)


@patch('app.processors.l1.l1_manager.L1Manager._store_l1_data')
@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_generate_l1_from_l0_no_notes(mock_extract, mock_store, l1_manager, mock_postgres_adapter):
    """Test L1 generation with no valid notes."""
    user_id = "test_user"
    
    # Set up the extract notes mock to return empty lists
    mock_extract.return_value = ([], [])
    
    # Make mock_store return None to handle the early return case
    mock_store.return_value = None
    
    # Test the method
    result = l1_manager.generate_l1_from_l0(user_id)
    
    # Check calls to dependencies
    mock_extract.assert_called_once_with(user_id)
    
    # Check result
    assert result is None or (isinstance(result, L1GenerationResult) and result.status == "failed")


@patch('app.processors.l1.l1_manager.L1Manager._store_l1_data')
@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_generate_l1_from_l0_exception_in_generation(
    mock_extract, 
    mock_store, 
    l1_manager, 
    mock_postgres_adapter, 
    mock_topics_generator, 
    sample_notes
):
    """Test handling exception during L1 generation."""
    user_id = "test_user"
    
    # Set up the extract notes mock
    mock_extract.return_value = (sample_notes, [{"memoryId": note.id} for note in sample_notes])
    
    # Make the generator raise an exception
    mock_topics_generator.generate_topics_for_shades.side_effect = Exception("Test error")
    
    # Test the method
    try:
        result = l1_manager.generate_l1_from_l0(user_id)
        # If we get here, the exception was caught and handled within the method
        assert result is None or (isinstance(result, L1GenerationResult) and result.status == "failed")
    except Exception as e:
        # The exception was re-raised, which is also an acceptable implementation
        assert str(e) == "Test error"


def test_extract_notes_from_l0(l1_manager):
    """Test extracting notes from L0."""
    # This requires a more complex setup - mocking database queries
    # We'll create a basic test for now that just verifies the method exists
    l1_manager.postgres_adapter.get_documents_with_l0.return_value = []
    
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
    shades = [{
        "id": "shade_0",
        "name": "Test Shade",
        "summary": "Test summary",
        "confidence": 0.9
    }]
    
    # Should not raise an exception
    l1_manager._store_l1_data(user_id, bio, clusters, chunk_topics, shades)
    
    # Check that the postgres adapter was called to get the latest version
    mock_postgres_adapter.get_latest_version.assert_called_once_with(user_id)


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