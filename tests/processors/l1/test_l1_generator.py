import pytest
from unittest.mock import MagicMock, patch

from app.processors.l1.l1_generator import L1Generator, MergeShadeResult
from app.models.l1.bio import Bio
from app.models.l1.shade import L1Shade


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
    mock.generate_shade_for_cluster.return_value = L1Shade(
        id="test_shade",
        user_id="test_user",
        name="Test Shade",
        summary="Test summary",
        confidence=0.9,
        s3_path="test/path"
    )
    mock.merge_shades.return_value = [
        {
            "id": "merged_shade",
            "name": "Merged Shade",
            "summary": "Merged summary",
            "confidence": 0.9
        }
    ]
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


def test_init():
    """Test L1Generator initialization."""
    generator = L1Generator()
    assert hasattr(generator, 'topics_generator')
    assert hasattr(generator, 'shade_generator')
    assert hasattr(generator, 'biography_generator')


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


def test_gen_shade_for_cluster(l1_generator, mock_shade_generator, sample_notes):
    """Test delegating to shade generator."""
    user_id = "test_user"
    old_shades = []
    cluster_notes = sample_notes[:2]
    memory_list = []
    
    result = l1_generator.gen_shade_for_cluster(
        user_id,
        old_shades,
        cluster_notes,
        memory_list
    )
    
    # Check that the shade generator was called
    mock_shade_generator.generate_shade_for_cluster.assert_called_once_with(
        user_id,
        old_shades,
        cluster_notes,
        memory_list
    )
    
    # Check result
    assert isinstance(result, L1Shade)
    assert result.name == "Test Shade"


def test_merge_shades(l1_generator, mock_shade_generator, sample_shades):
    """Test delegating to shade generator for merging."""
    user_id = "test_user"
    
    result = l1_generator.merge_shades(user_id, sample_shades)
    
    # Check that the shade generator was called
    mock_shade_generator.merge_shades.assert_called_once_with(user_id, sample_shades)
    
    # Check result
    assert isinstance(result, MergeShadeResult)
    assert result.success
    assert len(result.merge_shade_list) == 1
    assert result.merge_shade_list[0]["name"] == "Merged Shade"


def test_merge_shades_empty(l1_generator, mock_shade_generator):
    """Test merging with empty shades list."""
    user_id = "test_user"
    
    # Make the mock return empty list
    mock_shade_generator.merge_shades.return_value = []
    
    result = l1_generator.merge_shades(user_id, [])
    
    # Check result
    assert isinstance(result, MergeShadeResult)
    assert not result.success
    assert len(result.merge_shade_list) == 0


def test_merge_shades_single(l1_generator, mock_shade_generator, sample_shades):
    """Test merging with a single shade."""
    user_id = "test_user"
    single_shade = [sample_shades[0]]
    
    # Configure the mock for single shade case
    shade_dict = sample_shades[0].to_dict()
    mock_shade_generator.merge_shades.return_value = [shade_dict]
    
    result = l1_generator.merge_shades(user_id, single_shade)
    
    # Check result
    assert isinstance(result, MergeShadeResult)
    assert result.success
    assert len(result.merge_shade_list) == 1
    assert result.merge_shade_list[0] == shade_dict


def test_merge_shades_exception(l1_generator, mock_shade_generator):
    """Test handling exceptions during shade merging."""
    user_id = "test_user"
    
    # Make the mock raise an exception
    mock_shade_generator.merge_shades.side_effect = Exception("Test error")
    
    result = l1_generator.merge_shades(user_id, [])
    
    # Check result
    assert isinstance(result, MergeShadeResult)
    assert not result.success
    assert len(result.merge_shade_list) == 0


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