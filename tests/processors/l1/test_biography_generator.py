import pytest
from unittest.mock import MagicMock, patch
import json

from app.processors.l1.biography_generator import BiographyGenerator
from app.models.l1.bio import Bio


@pytest.fixture
def biography_generator(mock_llm_service, mock_wasabi_adapter):
    """Return a BiographyGenerator instance with mock dependencies."""
    return BiographyGenerator(
        llm_service=mock_llm_service,
        wasabi_adapter=mock_wasabi_adapter
    )


def test_init():
    """Test BiographyGenerator initialization."""
    generator = BiographyGenerator()
    assert hasattr(generator, 'llm_service')
    assert hasattr(generator, 'wasabi_adapter')


def test_generate_global_biography(biography_generator, sample_bio, sample_clusters):
    """Test generating a global biography."""
    result = biography_generator.generate_global_biography(
        user_id="test_user",
        old_profile=sample_bio,
        cluster_list=sample_clusters.get("clusterList", [])
    )
    
    # Check that we have a non-empty result
    assert result is not None
    assert isinstance(result, Bio)
    
    # Check bio properties
    assert result.content_third_view == "They are a test user."  # From mock LLM response
    assert result.summary_third_view == "Test user."  # From mock LLM response
    assert result.content_first_view == "I am a test user."  # From mock perspective shift
    assert result.shades_list == sample_bio.shades_list


def test_generate_status_biography(biography_generator, sample_bio):
    """Test generating a status biography."""
    recent_documents = [
        {
            "title": "Recent Doc 1",
            "content": "This is a recent document",
            "created_at": "2023-05-01T12:00:00"
        },
        {
            "title": "Recent Doc 2",
            "content": "This is another recent document",
            "created_at": "2023-05-02T12:00:00"
        }
    ]
    
    result = biography_generator.generate_status_biography(
        user_id="test_user",
        recent_documents=recent_documents,
        old_bio=sample_bio
    )
    
    # Check that we have a non-empty result
    assert result is not None
    assert isinstance(result, Bio)
    
    # Check bio properties
    assert result.content_third_view == "They are a test user."  # From mock LLM response
    assert result.summary_third_view == "Test user."  # From mock LLM response
    assert result.content_first_view == "I am a test user."  # From mock perspective shift


def test_generate_perspective_shifts(biography_generator, sample_bio):
    """Test generating first and second person perspectives."""
    # Create a test bio with only third-person content
    test_bio = Bio(
        content_third_view="They are a test user with interests in programming.",
        summary_third_view="They like programming."
    )
    
    result = biography_generator._generate_perspective_shifts(test_bio)
    
    # Check that first and second person views were generated
    assert result.content_first_view == "I am a test user."  # From mock perspective shift
    assert result.summary_first_view == "I am a test user."  # From mock perspective shift
    assert result.content_second_view == "I am a test user."  # From mock perspective shift
    assert result.summary_second_view == "I am a test user."  # From mock perspective shift


def test_shift_perspective(biography_generator):
    """Test shifting the perspective of a text."""
    original_text = "They are a software developer with 5 years of experience."
    
    # Test shifting to first person
    first_person = biography_generator._shift_perspective(
        original_text, "third-person", "first-person"
    )
    assert first_person == "I am a test user."  # From mock LLM response
    
    # Test shifting to second person
    second_person = biography_generator._shift_perspective(
        original_text, "third-person", "second-person"
    )
    assert second_person == "I am a test user."  # From mock LLM response


def test_format_shades_for_prompt(biography_generator):
    """Test formatting shades for inclusion in LLM prompt."""
    shades_list = [
        {"name": "Shade 1", "summary": "Summary of shade 1"},
        {"name": "Shade 2", "summary": "Summary of shade 2"}
    ]
    
    result = biography_generator._format_shades_for_prompt(shades_list)
    
    # Check format
    assert "Shade 1" in result
    assert "Summary:" in result
    assert "Summary of shade 1" in result
    assert "Shade 2" in result


def test_format_clusters_for_prompt(biography_generator):
    """Test formatting clusters for inclusion in LLM prompt."""
    cluster_list = [
        {"topic": "Topic 1", "tags": ["tag1", "tag2"]},
        {"topic": "Topic 2", "tags": ["tag3", "tag4"]}
    ]
    
    result = biography_generator._format_clusters_for_prompt(cluster_list)
    
    # Check format
    assert "Cluster 1" in result
    assert "Topic 1" in result
    assert "Tags:" in result
    assert "tag1, tag2" in result
    assert "Cluster 2" in result


def test_format_documents_for_prompt(biography_generator):
    """Test formatting documents for inclusion in LLM prompt."""
    documents = [
        {
            "title": "Doc 1",
            "content": "Content of document 1",
            "created_at": "2023-05-01T12:00:00"
        },
        {
            "title": "Doc 2",
            "content": "Content of document 2",
            "created_at": "2023-05-02T12:00:00"
        }
    ]
    
    result = biography_generator._format_documents_for_prompt(documents)
    
    # Check format
    assert "Document 1" in result
    assert "Doc 1" in result
    assert "Content:" in result
    assert "Content of document 1" in result
    assert "Document 2" in result


def test_format_documents_for_prompt_empty(biography_generator):
    """Test formatting empty documents list for prompt."""
    result = biography_generator._format_documents_for_prompt([])
    
    assert result == "No recent documents available."


def test_parse_bio_response(biography_generator):
    """Test parsing biography generation response."""
    content = '{"content_third_view": "They are a software developer.", "summary_third_view": "Software developer.", "confidence": 0.9}'
    
    result = biography_generator._parse_bio_response(content)
    
    # Check parsing
    assert result["content_third_view"] == "They are a software developer."
    assert result["summary_third_view"] == "Software developer."
    assert result["confidence"] == 0.9
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    result = biography_generator._parse_bio_response(malformed)
    assert result == {}


def test_parse_status_bio_response(biography_generator):
    """Test parsing status biography response."""
    content = '{"content_third_view": "They recently completed a project.", "summary_third_view": "Completed project."}'
    
    result = biography_generator._parse_status_bio_response(content)
    
    # Check parsing
    assert result["content_third_view"] == "They recently completed a project."
    assert result["summary_third_view"] == "Completed project."
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    result = biography_generator._parse_status_bio_response(malformed)
    assert result == {}


def test_store_bio_data(biography_generator, sample_bio, mock_wasabi_adapter):
    """Test storing biography data in Wasabi."""
    bio_data = sample_bio.to_dict()
    
    result = biography_generator._store_bio_data("test_user", bio_data, "global")
    
    # Check S3 path format
    assert result.startswith("l1/bios/test_user/global/")
    assert result.endswith(".json")
    
    # Verify Wasabi adapter was called correctly
    mock_wasabi_adapter.store_json.assert_called_once()
    call_args = mock_wasabi_adapter.store_json.call_args[0]
    
    # Check args to store_json
    assert call_args[0] == "test_user"  # user_id
    assert call_args[1] == result  # s3_path
    assert "bio" in call_args[2]  # complete_data
    assert call_args[2]["bio"] == bio_data 