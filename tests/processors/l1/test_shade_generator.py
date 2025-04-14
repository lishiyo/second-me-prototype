import pytest
from unittest.mock import MagicMock, patch
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from app.processors.l1.shade_generator import ShadeGenerator
from app.models.l1.shade import Shade as L1Shade

# Mock Note class for testing
@dataclass
class Note:
    """Mock Note class for testing."""
    id: str
    title: str
    content: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for testing."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "user_id": self.user_id,
            "metadata": self.metadata
        }


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


def test_merge_shades_multiple(shade_generator, sample_shades, mock_merge_llm_service):
    """Test merging multiple shades."""
    # Temporarily replace the llm_service with our merge-specific mock
    original_llm_service = shade_generator.llm_service
    shade_generator.llm_service = mock_merge_llm_service
    
    # Add timelines to metadata for testing
    for shade in sample_shades:
        if not hasattr(shade, "metadata"):
            shade.metadata = {}
        shade.metadata["timelines"] = [
            {"createTime": "2023-05-15", "description": "Event description", "refId": f"doc-{shade.id}"}
        ]
    
    try:
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
        assert "metadata" in merged_shade
        assert "timelines" in merged_shade["metadata"]
        assert len(merged_shade["metadata"]["timelines"]) > 0
        assert merged_shade["name"] == "Merged Shade"  # From mock LLM response
    finally:
        # Restore the original llm_service
        shade_generator.llm_service = original_llm_service


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
    content = '{"name": "Test Shade", "summary": "This is a test summary", "confidence": 0.85, "timelines": [{"createTime": "2023-05-15", "description": "Event description", "refId": "doc1"}]}'
    
    result = shade_generator._parse_shade_response(content)
    
    # Check parsing
    assert result["name"] == "Test Shade"
    assert result["summary"] == "This is a test summary"
    assert result["confidence"] == 0.85
    assert "timelines" in result
    assert len(result["timelines"]) == 1
    assert result["timelines"][0]["createTime"] == "2023-05-15"
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    result = shade_generator._parse_shade_response(malformed)
    assert result == {}


def test_parse_merged_shades_response(shade_generator):
    """Test parsing merged shades response."""
    content = '''[
        {
            "name": "Merged Shade 1", 
            "summary": "Summary 1", 
            "confidence": 0.9,
            "timelines": [
                {"createTime": "2023-05-15", "description": "Event 1", "refId": "doc1"}
            ]
        }, 
        {
            "name": "Merged Shade 2", 
            "summary": "Summary 2", 
            "confidence": 0.8,
            "timelines": []
        }
    ]'''
    
    result = shade_generator._parse_merged_shades_response(content)
    
    # Check parsing
    assert len(result) == 2
    assert result[0]["name"] == "Merged Shade 1"
    assert result[0]["summary"] == "Summary 1"
    assert result[0]["confidence"] == 0.9
    assert "timelines" in result[0]
    assert len(result[0]["timelines"]) == 1
    assert result[1]["name"] == "Merged Shade 2"
    assert "timelines" in result[1]
    assert len(result[1]["timelines"]) == 0
    
    # Test with malformed JSON
    malformed = "Not a JSON [missing: quotes]"
    result = shade_generator._parse_merged_shades_response(malformed)
    assert result == []


def test_store_shade_data(shade_generator, sample_notes, mock_wasabi_adapter):
    """Test storing shade data in Wasabi."""
    shade_data = {
        "name": "Test Shade",
        "summary": "Test summary",
        "confidence": 0.85,
        "timelines": [
            {"createTime": "2023-05-15", "description": "Event description", "refId": "doc1"}
        ]
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
        "center_embedding": None,
        "timelines": shade_data["timelines"]
    }
    assert call_args[1]["shade"] == expected_shade
    assert "notes" in call_args[1]
    assert len(call_args[1]["notes"]) == 2


def test_store_merged_shade_data(shade_generator, mock_wasabi_adapter):
    """Test storing merged shade data in Wasabi."""
    shade_data = {
        "name": "Merged Shade",
        "summary": "Merged summary",
        "confidence": 0.9,
        "timelines": [
            {"createTime": "2023-05-15", "description": "Event description", "refId": "doc1"}
        ]
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
        "center_embedding": None,
        "timelines": shade_data["timelines"]
    }
    assert call_args[1]["shade"] == expected_shade


@pytest.fixture
def sample_shade(sample_notes):
    """Return a sample shade for testing."""
    return L1Shade(
        id="test-shade-id",
        name="Test Shade",
        user_id="test-user",
        summary="This is a test shade summary",
        confidence=0.85,
        metadata={
            "timelines": [
                {"createTime": "2023-05-15", "description": "Event description", "refId": "doc1"}
            ]
        }
    )


def test_improve_shade(shade_generator, sample_shade, sample_notes, mock_wasabi_adapter):
    """Test improving a shade with new notes."""
    # Mock the LLM response for improvement
    mock_content = '''{
        "improved_name": "Improved Test Shade",
        "improved_summary": "This is an improved test shade summary",
        "improved_confidence": 0.9,
        "new_timelines": [
            {"createTime": "2023-06-20", "description": "New event", "refId": "doc3"}
        ]
    }'''
    
    shade_generator.llm_service.chat_completion.return_value = {
        "choices": [{"message": {"content": mock_content}}]
    }
    
    # Call the method
    result = shade_generator.improve_shade(
        user_id="test-user",
        old_shade=sample_shade,
        new_notes=sample_notes[:1]
    )
    
    # Check results
    assert result is not None
    assert result.name == "Improved Test Shade"
    assert result.summary == "This is an improved test shade summary"
    assert result.confidence == 0.9
    assert "timelines" in result.metadata
    assert len(result.metadata["timelines"]) == 2  # Original + new timeline
    
    # Check that Wasabi was called to store the improved shade
    mock_wasabi_adapter.store_json.assert_called_once()
    call_args = mock_wasabi_adapter.store_json.call_args[0]
    assert call_args[0].startswith("l1/improved_shades/test-user/")
    assert "shade" in call_args[1]
    assert "new_notes" in call_args[1]


def test_format_shade_for_improvement(shade_generator, sample_shade):
    """Test formatting a shade for improvement prompt."""
    result = shade_generator._format_shade_for_improvement(sample_shade)
    
    # Check content
    assert "Name: Test Shade" in result
    assert "Summary: This is a test shade summary" in result
    assert "Confidence: 0.85" in result
    assert "Timelines:" in result
    assert "2023-05-15" in result
    assert "Event description" in result


def test_parse_improved_shade_response(shade_generator):
    """Test parsing improved shade response."""
    content = '''{
        "improved_name": "Improved Shade",
        "improved_summary": "Improved summary",
        "improved_confidence": 0.9,
        "new_timelines": [
            {"createTime": "2023-06-20", "description": "New event", "refId": "doc3"}
        ]
    }'''
    
    result = shade_generator._parse_improved_shade_response(content)
    
    # Check parsing
    assert result["improved_name"] == "Improved Shade"
    assert result["improved_summary"] == "Improved summary"
    assert result["improved_confidence"] == 0.9
    assert "new_timelines" in result
    assert len(result["new_timelines"]) == 1
    assert result["new_timelines"][0]["createTime"] == "2023-06-20"
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    result = shade_generator._parse_improved_shade_response(malformed)
    assert result == {} 


@pytest.fixture
def mock_llm_service():
    """Mock the LLM service."""
    service = MagicMock()
    # Mock the chat_completion method
    service.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '''
                    {
                        "name": "Test Shade",
                        "summary": "This is a test shade summary",
                        "confidence": 0.85,
                        "timelines": [
                            {"createTime": "2023-05-15", "description": "Test Event", "refId": "doc1"}
                        ]
                    }
                    '''
                }
            }
        ]
    }
    return service

@pytest.fixture
def mock_merge_llm_service():
    """Mock the LLM service for shade merging."""
    service = MagicMock()
    # Mock the chat_completion method for merging
    service.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '''
                    [
                        {
                            "name": "Merged Shade",
                            "summary": "This is a merged shade summary",
                            "confidence": 0.9,
                            "timelines": [
                                {"createTime": "2023-05-15", "description": "Merged Event", "refId": "doc1"}
                            ]
                        }
                    ]
                    '''
                }
            }
        ]
    }
    return service

@pytest.fixture
def mock_wasabi_adapter():
    """Mock the Wasabi storage adapter."""
    adapter = MagicMock()
    adapter.store_json.return_value = True
    return adapter

@pytest.fixture
def sample_notes():
    """Return sample notes for testing."""
    return [
        Note(
            id="note1",
            title="Note 1",
            content="This is the content of note 1",
            user_id="test-user"
        ),
        Note(
            id="note2",
            title="Note 2",
            content="This is the content of note 2",
            user_id="test-user"
        ),
        Note(
            id="note3",
            title="Note 3",
            content="This is the content of note 3",
            user_id="test-user"
        )
    ]

@pytest.fixture
def sample_shades():
    """Return sample shades for testing."""
    return [
        L1Shade(
            id="shade1",
            name="Test Shade 1",
            user_id="test-user",
            summary="This is test shade 1",
            content="Detail about test shade 1",
            confidence=0.8,
            metadata={}
        ),
        L1Shade(
            id="shade2",
            name="Test Shade 2",
            user_id="test-user",
            summary="This is test shade 2",
            content="Detail about test shade 2",
            confidence=0.9,
            metadata={}
        )
    ] 