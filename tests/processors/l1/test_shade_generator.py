import pytest
from unittest.mock import MagicMock, patch
import json

from app.processors.l1.shade_generator import ShadeGenerator
from app.models.l1.shade import Shade as L1Shade


@pytest.fixture
def shade_generator(mock_llm_service, mock_wasabi_adapter):
    """Return a ShadeGenerator instance with mock dependencies."""
    return ShadeGenerator(
        llm_service=mock_llm_service,
        wasabi_adapter=mock_wasabi_adapter
    )


def test_init():
    """Test ShadeGenerator initialization."""
    generator = ShadeGenerator()
    assert hasattr(generator, 'llm_service')
    assert hasattr(generator, 'wasabi_adapter')


def test_generate_shade_for_cluster_empty_notes(shade_generator):
    """Test generating a shade with empty notes list."""
    result = shade_generator.generate_shade_for_cluster(
        user_id="test_user",
        old_shades=[],
        cluster_notes=[],
        memory_list=[]
    )
    assert result is None


def test_generate_shade_for_cluster(shade_generator, sample_notes):
    """Test generating a shade for a cluster of notes."""
    result = shade_generator.generate_shade_for_cluster(
        user_id="test_user",
        old_shades=[],
        cluster_notes=sample_notes[:2],
        memory_list=[]
    )
    
    # Check that we have a non-empty result
    assert result is not None
    assert isinstance(result, L1Shade)
    
    # Check shade properties
    assert result.user_id == "test_user"
    assert result.name == "Test Shade"  # From mock LLM response
    assert result.summary == "This is a test shade summary"  # From mock LLM response
    assert result.confidence == 0.85  # From mock LLM response
    assert result.s3_path.startswith("l1/shades/test_user/")
    assert result.s3_path.endswith(".json")


def test_merge_shades_empty(shade_generator):
    """Test merging shades with empty input."""
    result = shade_generator.merge_shades("test_user", [])
    
    # Should return an empty list
    assert result == []


def test_merge_shades_single(shade_generator, sample_shades):
    """Test merging a single shade (should return as-is)."""
    result = shade_generator.merge_shades("test_user", [sample_shades[0]])
    
    # Should return the single shade
    assert len(result) == 1
    assert result[0]["id"] == sample_shades[0].id
    assert result[0]["name"] == sample_shades[0].name


def test_merge_shades_multiple(shade_generator, sample_shades):
    """Test merging multiple shades."""
    result = shade_generator.merge_shades("test_user", sample_shades)
    
    # Check the result
    assert result is not None
    assert len(result) > 0  # At least one merged shade
    
    # Check properties of the merged shade
    merged_shade = result[0]
    assert "id" in merged_shade
    assert "name" in merged_shade
    assert "summary" in merged_shade
    assert "confidence" in merged_shade
    assert merged_shade["name"] == "Merged Shade"  # From mock LLM response


def test_format_notes_for_prompt(shade_generator, sample_notes):
    """Test formatting notes for inclusion in LLM prompt."""
    result = shade_generator._format_notes_for_prompt(sample_notes[:2])
    
    # Check format
    assert "Document 1" in result
    assert "Content:" in result
    assert sample_notes[0].title in result
    assert sample_notes[0].content[:50] in result  # Check start of content


def test_format_shades_for_prompt(shade_generator, sample_shades):
    """Test formatting shades for inclusion in LLM prompt."""
    result = shade_generator._format_shades_for_prompt(sample_shades[:2])
    
    # Check format
    assert "Shade 1" in result
    assert "Summary:" in result
    assert sample_shades[0].name in result
    assert sample_shades[0].summary in result


def test_parse_shade_response(shade_generator):
    """Test parsing shade generation response."""
    content = '{"name": "Test Shade", "summary": "This is a test summary", "confidence": 0.85}'
    
    result = shade_generator._parse_shade_response(content)
    
    # Check parsing
    assert result["name"] == "Test Shade"
    assert result["summary"] == "This is a test summary"
    assert result["confidence"] == 0.85
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    result = shade_generator._parse_shade_response(malformed)
    assert result == {}


def test_parse_merged_shades_response(shade_generator):
    """Test parsing merged shades response."""
    content = '[{"name": "Merged Shade 1", "summary": "Summary 1", "confidence": 0.9}, {"name": "Merged Shade 2", "summary": "Summary 2", "confidence": 0.8}]'
    
    result = shade_generator._parse_merged_shades_response(content)
    
    # Check parsing
    assert len(result) == 2
    assert result[0]["name"] == "Merged Shade 1"
    assert result[0]["summary"] == "Summary 1"
    assert result[0]["confidence"] == 0.9
    assert result[1]["name"] == "Merged Shade 2"
    
    # Test with malformed JSON
    malformed = "Not a JSON [missing: quotes]"
    result = shade_generator._parse_merged_shades_response(malformed)
    assert result == []


def test_store_shade_data(shade_generator, sample_notes, mock_wasabi_adapter):
    """Test storing shade data in Wasabi."""
    shade_data = {
        "name": "Test Shade",
        "summary": "Test summary",
        "confidence": 0.85
    }
    
    result = shade_generator._store_shade_data("test_user", shade_data, sample_notes[:2])
    
    # Check S3 path format
    assert result.startswith("l1/shades/test_user/")
    assert result.endswith(".json")
    
    # Verify Wasabi adapter was called correctly
    mock_wasabi_adapter.store_json.assert_called_once()
    call_args = mock_wasabi_adapter.store_json.call_args[0]
    
    # Check args to store_json
    assert call_args[0] == result  # s3_path
    assert "shade" in call_args[1]  # complete_data
    
    # Check expected structure matches our transformed structure
    expected_shade = {
        "name": shade_data["name"],
        "summary": shade_data["summary"],
        "confidence": shade_data["confidence"],
        "center_embedding": None
    }
    assert call_args[1]["shade"] == expected_shade
    assert "notes" in call_args[1]
    assert len(call_args[1]["notes"]) == 2


def test_store_merged_shade_data(shade_generator, mock_wasabi_adapter):
    """Test storing merged shade data in Wasabi."""
    shade_data = {
        "name": "Merged Shade",
        "summary": "Merged summary",
        "confidence": 0.9
    }
    
    result = shade_generator._store_merged_shade_data("test_user", shade_data)
    
    # Check S3 path format
    assert result.startswith("l1/merged_shades/test_user/")
    assert result.endswith(".json")
    
    # Verify Wasabi adapter was called correctly
    mock_wasabi_adapter.store_json.assert_called_once()
    call_args = mock_wasabi_adapter.store_json.call_args[0]
    
    # Check args to store_json
    assert call_args[0] == result  # s3_path
    assert "shade" in call_args[1]  # complete_data
    
    # Check expected structure matches our transformed structure
    expected_shade = {
        "name": shade_data["name"],
        "summary": shade_data["summary"],
        "confidence": shade_data["confidence"],
        "center_embedding": None
    }
    assert call_args[1]["shade"] == expected_shade 