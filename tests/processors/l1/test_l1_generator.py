import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.processors.l1.l1_generator import L1Generator, MergeShadeResult
from app.models.l1.bio import Bio
from app.models.l1.shade import L1Shade
from app.models.l1.note import Note


@pytest.fixture
def mock_topics_generator():
    """Return a mock TopicsGenerator."""
    mock = MagicMock()
    mock.generate_topics.return_value = {"test_cluster": {"topic": "Test Topic"}}
    mock.generate_topics_for_shades.return_value = {
        "clusterList": [{"clusterId": "test_cluster", "topic": "Test Topic"}],
        "outlierMemoryList": []
    }
    return mock


@pytest.fixture
def mock_shade_generator():
    """Return a mock ShadeGenerator."""
    mock = MagicMock()
    # Configure generate_shade_for_cluster to return a shade with timeline data
    mock.generate_shade_for_cluster.return_value = L1Shade(
        id="test_shade",
        user_id="test_user",
        name="Test Shade",
        summary="Test summary",
        confidence=0.9,
        s3_path="test/path",
        metadata={
            "center_embedding": [0.1, 0.2, 0.3],
            "timelines": [
                {"createTime": "2023-01-01", "description": "Test event", "refId": "1"}
            ]
        }
    )
    # Configure merge_shades to return shades with timeline data
    mock.merge_shades.return_value = [
        {
            "id": "merged_shade",
            "name": "Merged Shade",
            "summary": "Merged summary",
            "confidence": 0.9,
            "timelines": [
                {"createTime": "2023-01-01", "description": "Test event", "refId": "1"}
            ]
        }
    ]
    # Configure improve_shade to return an improved shade
    improved_shade = L1Shade(
        id="test_shade",
        user_id="test_user",
        name="Improved Shade",
        summary="Improved summary",
        confidence=0.95,
        s3_path="test/improved_path",
        metadata={
            "center_embedding": [0.1, 0.2, 0.3],
            "timelines": [
                {"createTime": "2023-01-01", "description": "Test event", "refId": "1"},
                {"createTime": "2023-01-02", "description": "New event", "refId": "2"}
            ]
        }
    )
    mock.improve_shade.return_value = improved_shade
    return mock


@pytest.fixture
def mock_biography_generator():
    """Return a mock BiographyGenerator."""
    mock = MagicMock()
    mock.generate_global_biography.return_value = Bio(
        content_third_view="Test bio content",
        summary_third_view="Test bio summary"
    )
    mock.generate_status_biography.return_value = Bio(
        content_third_view="Test status content",
        summary_third_view="Test status summary"
    )
    return mock


@pytest.fixture
def l1_generator(mock_topics_generator, mock_shade_generator, mock_biography_generator):
    """Return an L1Generator with mock dependencies."""
    return L1Generator(
        topics_generator=mock_topics_generator,
        shade_generator=mock_shade_generator,
        biography_generator=mock_biography_generator
    )


@pytest.fixture
def sample_notes():
    """Return sample notes for testing."""
    return [
        Note(
            id="1", 
            title="Note 1", 
            content="Content 1", 
            create_time=datetime.now(),
            metadata={"user_id": "test_user"}
        ),
        Note(
            id="2", 
            title="Note 2", 
            content="Content 2", 
            create_time=datetime.now(),
            metadata={"user_id": "test_user"}
        )
    ]


@pytest.fixture
def sample_shades():
    """Return sample shades for testing."""
    return [
        L1Shade(
            id="shade1",
            user_id="test_user",
            name="Shade 1",
            summary="Summary 1",
            confidence=0.8,
            metadata={
                "center_embedding": [0.1, 0.2, 0.3],
                "cluster_size": 2,
                "timelines": [
                    {"createTime": "2023-01-01", "description": "Test event 1", "refId": "1"}
                ]
            }
        ),
        L1Shade(
            id="shade2",
            user_id="test_user",
            name="Shade 2",
            summary="Summary 2",
            confidence=0.7,
            metadata={
                "center_embedding": [0.2, 0.3, 0.4],
                "cluster_size": 3,
                "timelines": [
                    {"createTime": "2023-01-02", "description": "Test event 2", "refId": "2"}
                ]
            }
        )
    ]


@pytest.fixture
def sample_bio():
    """Return a sample biography for testing."""
    return Bio(
        content_third_view="Sample bio content",
        summary_third_view="Sample bio summary"
    )

def test_generate_topics(l1_generator, mock_topics_generator, sample_notes):
    """Test delegating to topics generator."""
    result = l1_generator.generate_topics(sample_notes)
    
    # Check that the topics generator was called
    mock_topics_generator.generate_topics.assert_called_once_with(sample_notes)
    
    # Check result
    assert result == {"test_cluster": {"topic": "Test Topic"}}


def test_gen_topics_for_shades(l1_generator, mock_topics_generator):
    """Test delegating to topics generator for shades."""
    user_id = "test_user"
    old_cluster_list = []
    old_outlier_memory_list = []
    new_memory_list = [{"memoryId": "test_memory"}]
    
    result = l1_generator.gen_topics_for_shades(
        user_id,
        old_cluster_list,
        old_outlier_memory_list,
        new_memory_list
    )
    
    # Check that the topics generator was called
    mock_topics_generator.generate_topics_for_shades.assert_called_once_with(
        old_cluster_list,
        old_outlier_memory_list,
        new_memory_list
    )
    
    # Check result
    assert result == {
        "clusterList": [{"clusterId": "test_cluster", "topic": "Test Topic"}],
        "outlierMemoryList": []
    }

def test_gen_global_biography(l1_generator, mock_biography_generator, sample_bio):
    """Test delegating to biography generator."""
    user_id = "test_user"
    old_profile = sample_bio
    cluster_list = [{"topic": "Test Topic"}]
    
    result = l1_generator.gen_global_biography(user_id, old_profile, cluster_list)
    
    # Check that the biography generator was called
    mock_biography_generator.generate_global_biography.assert_called_once_with(
        user_id,
        old_profile,
        cluster_list
    )
    
    # Check result
    assert isinstance(result, Bio)
    assert result.content_third_view == "Test bio content"
    assert result.summary_third_view == "Test bio summary"


def test_gen_status_biography(l1_generator, mock_biography_generator, sample_bio):
    """Test delegating to biography generator for status biography."""
    user_id = "test_user"
    recent_documents = [{"title": "Recent Doc"}]
    old_bio = sample_bio
    
    result = l1_generator.gen_status_biography(user_id, recent_documents, old_bio)
    
    # Check that the biography generator was called
    mock_biography_generator.generate_status_biography.assert_called_once_with(
        user_id,
        recent_documents,
        old_bio
    )
    
    # Check result
    assert isinstance(result, Bio)
    assert result.content_third_view == "Test status content"
    assert result.summary_third_view == "Test status summary" 