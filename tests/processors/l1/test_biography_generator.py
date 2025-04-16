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


def test_init(mock_llm_service, mock_wasabi_adapter):
    """Test BiographyGenerator initialization."""
    generator = BiographyGenerator(
        llm_service=mock_llm_service,
        wasabi_adapter=mock_wasabi_adapter
    )
    assert hasattr(generator, 'llm_service')
    assert hasattr(generator, 'wasabi_adapter')
    assert generator.llm_service == mock_llm_service
    assert generator.wasabi_adapter == mock_wasabi_adapter
    assert generator.bio_model_params['temperature'] == 0
    assert generator.bio_model_params['max_tokens'] == 2000


def test_generate_global_biography(biography_generator, sample_bio, sample_clusters):
    """Test generating a global biography."""
    # Patch the internal method to avoid LLM calls
    with patch.object(biography_generator, '_call_llm_with_retry') as mock_llm:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Comprehensive analysis of test user"
        mock_llm.return_value = mock_response
        
        # Also patch perspective shifts to avoid additional LLM calls
        with patch.object(biography_generator, '_generate_perspective_shifts') as mock_shift:
            # Make mock_shift return the input bio with updated content
            def add_perspectives(bio):
                bio.content_third_view = "Comprehensive Analysis Report\nUser's Interests and Preferences\nConclusion"
                bio.content_first_view = "I am a test user."
                return bio
            mock_shift.side_effect = add_perspectives
            
            result = biography_generator.generate_global_biography(
                user_id="test_user",
                old_profile=sample_bio,
                cluster_list=sample_clusters.get("clusterList", [])
            )
    
    # Check that we have a non-empty result
    assert result is not None
    assert isinstance(result, Bio)
    
    # Check bio properties
    assert "Comprehensive Analysis Report" in result.content_third_view
    assert "Conclusion" in result.content_third_view
    assert result.content_first_view == "I am a test user."
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
    
    # Patch the internal methods to avoid LLM calls
    with patch.object(biography_generator, '_call_llm_with_retry') as mock_llm:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "content_third_view": "They are a test user.",
            "summary_third_view": "Test user.",
            "health_status": "They appear to be in good health."
        })
        mock_llm.return_value = mock_response
        
        # Also patch perspective shifts to avoid additional LLM calls
        with patch.object(biography_generator, '_generate_perspective_shifts') as mock_shift:
            # Make mock_shift return the input bio with updated content
            def add_perspectives(bio):
                bio.content_first_view = "I am a test user."
                return bio
            mock_shift.side_effect = add_perspectives
            
            result = biography_generator.generate_status_biography(
                user_id="test_user",
                recent_documents=recent_documents,
                old_bio=sample_bio
            )
    
    # Check that we have a non-empty result
    assert result is not None
    assert isinstance(result, Bio)
    
    # Check bio properties
    assert result.content_third_view == "They are a test user."
    assert result.summary_third_view == "Test user."
    assert result.health_status == "They appear to be in good health."
    assert result.content_first_view == "I am a test user."


def test_generate_perspective_shifts(biography_generator, sample_bio):
    """Test generating first and second person perspectives."""
    # Create a test bio with only third-person content
    test_bio = Bio(
        content_third_view="They are a test user with interests in programming.",
        summary_third_view="They like programming."
    )
    
    # Patch the internal methods to avoid LLM calls
    with patch.object(biography_generator, '_shift_perspective_to_second') as mock_second:
        def add_second_perspective(bio):
            bio.content_second_view = "You are a test user with interests in programming."
            bio.summary_second_view = "You like programming."
            return bio
        mock_second.side_effect = add_second_perspective
        
        with patch.object(biography_generator, '_shift_perspective_to_first') as mock_first:
            mock_first.return_value = "I am a test user."
            
            result = biography_generator._generate_perspective_shifts(test_bio)
    
    # Check that first and second person views were generated
    assert result.content_first_view == "I am a test user."
    assert result.summary_first_view == "I am a test user."
    assert result.content_second_view == "You are a test user with interests in programming."
    assert result.summary_second_view == "You like programming."


def test_shift_perspective_to_second(biography_generator, sample_bio):
    """Test shifting the biography to second-person perspective."""
    test_bio = Bio(
        content_third_view="They are a test user with interests in programming.",
        summary_third_view="They like programming."
    )
    
    # Patch the internal method to avoid LLM calls
    with patch.object(biography_generator, '_call_llm_with_retry') as mock_llm:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "You like programming."
        mock_llm.return_value = mock_response
        
        result = biography_generator._shift_perspective_to_second(test_bio)
    
    # Check second person perspective was generated
    assert result.summary_second_view == "You like programming."
    assert "User's Interests and Preferences" in result.content_second_view


def test_shift_perspective_to_first(biography_generator):
    """Test shifting a text from third-person to first-person."""
    original_text = "They are a software developer with 5 years of experience."
    
    # Patch the internal method to avoid LLM calls
    with patch.object(biography_generator, '_call_llm_with_retry') as mock_llm:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I am a software developer with 5 years of experience."
        mock_llm.return_value = mock_response
        
        # Test shifting to first person
        first_person = biography_generator._shift_perspective_to_first(
            original_text, "third-person", "first-person"
        )
    
    assert first_person == "I am a software developer with 5 years of experience."


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

def test_parse_status_bio_response(biography_generator):
    """Test parsing status biography response."""
    content = '{"content_third_view": "They recently completed a project.", "summary_third_view": "Completed project.", "health_status": "They appear to be in good health."}'
    
    result = biography_generator._parse_status_bio_response(content)
    
    # Check parsing
    assert result["content_third_view"] == "They recently completed a project."
    assert result["summary_third_view"] == "Completed project."
    assert result["health_status"] == "They appear to be in good health."
    
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
    assert call_args[0] == result  # s3_path
    assert "bio" in call_args[1]  # complete_data
    assert call_args[1]["bio"] == bio_data 