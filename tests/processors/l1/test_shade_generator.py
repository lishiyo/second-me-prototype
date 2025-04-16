import pytest
from unittest.mock import MagicMock, patch
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from app.processors.l1.shade_generator import ShadeGenerator
from app.models.l1.shade import L1Shade

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
    
    def to_str(self) -> str:
        """Format the note as a string for testing."""
        return f"Title: {self.title}\nContent: {self.content}\nID: {self.id}"


@pytest.fixture
def shade_generator(mock_llm_service, mock_wasabi_adapter):
    """Return a ShadeGenerator instance with mock dependencies."""
    return ShadeGenerator(
        llm_service=mock_llm_service,
        wasabi_adapter=mock_wasabi_adapter
    )


def test_init(mock_llm_service, mock_wasabi_adapter):
    """Test ShadeGenerator initialization."""
    generator = ShadeGenerator(
        llm_service=mock_llm_service,
        wasabi_adapter=mock_wasabi_adapter
    )
    assert hasattr(generator, 'llm_service')
    assert hasattr(generator, 'wasabi_adapter')


# def test_generate_shade_for_cluster_empty_notes(shade_generator):
#     """Test generating a shade with empty notes list."""
#     result = shade_generator.generate_shade_for_cluster(
#         user_id="test_user",
#         old_shades=[],
#         cluster_notes=[],
#         memory_list=[]
#     )
#     assert result is None


# def test_generate_shade_for_cluster(shade_generator, sample_notes):
#     """Test generating a shade for a cluster of notes."""
#     result = shade_generator.generate_shade_for_cluster(
#         user_id="test_user",
#         old_shades=[],
#         cluster_notes=sample_notes[:2],
#         memory_list=[]
#     )
    
#     # Check that we have a non-empty result
#     assert result is not None
#     assert isinstance(result, L1Shade)
    
#     # Check shade properties
#     assert result.user_id == "test_user"
#     assert result.name == "Test Shade"  # From mock LLM response
#     assert result.summary == "This is a test shade summary"  # From mock LLM response
#     assert result.confidence == 0.85  # From mock LLM response
#     assert result.s3_path.startswith("l1/shades/test_user/")
#     assert result.s3_path.endswith(".json")


def test_merge_shades_empty(shade_generator):
    """Test merging shades with empty input."""
    # Call with empty shade_info_list (should return None)
    result = shade_generator.generate_shade(
        user_id="test_user", 
        old_memory_list=[], 
        new_memory_list=[], 
        shade_info_list=[]
    )
    
    # Should return None
    assert result is None


def test_merge_shades_single(shade_generator, sample_shades, sample_notes):
    """Test generating a shade with a single existing shade."""
    # Set up a mock response for the LLM service
    shade_generator.llm_service.call_with_retry.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "improveDesc": "This is an improved description",
            "improveContent": "This is an improved content",
            "improveTimelines": []
        })))
    ]
    
    # Call generate_shade with a single shade - need both old_memory_list and shade_info_list
    result = shade_generator.generate_shade(
        user_id="test_user",
        old_memory_list=sample_notes,  # Need to provide old_memory_list to avoid errors
        new_memory_list=sample_notes,  # Need to provide new_memory_list as well
        shade_info_list=[sample_shades[0]]
    )
    
    # Should get a valid result
    assert result is not None
    assert result.id == sample_shades[0].id
    assert result.name == sample_shades[0].name


def test_merge_shades_multiple(shade_generator, sample_shades, mock_merge_llm_service, sample_notes):
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
    
    # Setup a mock response for the LLM for merge
    mock_merge_llm_service.call_with_retry.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "newInterestName": "Merged Shade",
            "newInterestDesc": "This is a merged shade",
            "newInterestContent": "This is merged content",
            "newInterestConfidence": 0.95,
            "newInterestAspect": "Merged aspect",
            "newInterestIcon": "icon-merged",
            "newInterestTimelines": []
        })))
    ]
    
    try:
        # Call generate_shade with multiple shades to trigger merging path
        result = shade_generator.generate_shade(
            user_id="test_user",
            old_memory_list=sample_notes,
            new_memory_list=sample_notes,
            shade_info_list=sample_shades
        )
        
        # Check the result
        assert result is not None
        assert result.name == "Merged Shade"
        assert result.desc_third_view == "This is a merged shade"
        assert result.content_third_view == "This is merged content"
        assert result.confidence == 0.95
        
    finally:
        # Restore the original llm_service
        shade_generator.llm_service = original_llm_service


def test_format_shades_for_prompt(shade_generator, sample_shades):
    """Test formatting shades for inclusion in LLM prompt."""
    result = shade_generator._format_shades_for_prompt(sample_shades[:2])
    
    # Check format
    assert "User Interest Domain 1 Analysis:" in result
    assert "**[Name]**: Test Shade 1" in result
    assert "**[Description]**:" in result
    assert "Description of test shade 1" in result


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
    
    # Test None handling
    content_with_none = '{"name": "Test Shade", "summary": null, "confidence": 0.85}'
    result = shade_generator._parse_shade_response(content_with_none)
    assert result["name"] == "Test Shade"
    assert result["summary"] == ""  # null should be handled in the function
    
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
    
    # Test with None values
    content_with_none = '''[{"name": "Merged Shade", "summary": null, "confidence": 0.9}]'''
    result = shade_generator._parse_merged_shades_response(content_with_none)
    assert len(result) == 1
    assert result[0]["name"] == "Merged Shade"
    assert result[0]["summary"] == ""  # null should be handled in the function
    
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
        desc_third_view="This is a test shade description",
        content_third_view="This is a test shade content",
        metadata={
            "timelines": [
                {"createTime": "2023-05-15", "description": "Event description", "refId": "doc1"}
            ]
        }
    )


def test_improve_shade(shade_generator, sample_shade, sample_notes):
    """Test improving a shade with new notes."""
    # Mock the LLM response for improvement
    shade_generator.llm_service.call_with_retry.return_value.choices = [
        MagicMock(message=MagicMock(content='''{
            "improveDesc": "This is an improved test shade description",
            "improveContent": "This is an improved test shade content",
            "improveTimelines": [
                {"createTime": "2023-06-20", "descThirdView": "New event", "refMemoryId": "doc3"}
            ]
        }'''))
    ]
    
    # Call the method
    result = shade_generator._improve_shade_process(
        user_id="test-user",
        old_shade=sample_shade,
        new_notes=sample_notes[:1]
    )
    
    # Check results
    assert result is not None
    assert result.name == sample_shade.name
    assert result.desc_third_view == "This is an improved test shade description"
    assert result.content_third_view == "This is an improved test shade content"
    assert "timelines" in result.metadata
    assert len(result._timelines) >= 1  # Should have at least the original timeline


def test_format_shade_for_improvement(shade_generator, sample_shade):
    """Test formatting a shade for improvement prompt."""
    result = shade_generator._format_shade_for_improvement(sample_shade)
    
    # Check content for specific patterns that match the to_str format
    assert "**[Name]**: Test Shade" in result
    assert "**[Description]**:" in result
    assert "**[Content]**:" in result
    assert "This is a test shade content" in result
    assert "**[Timelines]**:" in result
    assert "2023-05-15" in result
    assert "Event description" in result


def test_parse_improved_shade_response(shade_generator):
    """Test parsing improved shade response."""
    content = '''{
        "improveDesc": "Improved description",
        "improveContent": "Improved content",
        "improveTimelines": [
            {"createTime": "2023-06-20", "descThirdView": "New event", "refMemoryId": "doc3"}
        ]
    }'''
    
    result = shade_generator._parse_improved_shade_response(content)
    
    # Check parsing
    assert "improved_name" in result
    assert "improved_summary" in result
    assert "improved_confidence" in result
    assert "new_timelines" in result
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    result = shade_generator._parse_improved_shade_response(malformed)
    assert result == {}


@pytest.fixture
def mock_llm_service():
    """Mock the LLM service."""
    service = MagicMock()
    # Mock the call_with_retry method
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content='''
                {
                    "name": "Test Shade",
                    "summary": "This is a test shade summary",
                    "confidence": 0.85,
                    "timelines": [
                        {"createTime": "2023-05-15", "description": "Test Event", "refId": "doc1"}
                    ]
                }
                '''
            )
        )
    ]
    service.call_with_retry.return_value = response
    return service

@pytest.fixture
def mock_merge_llm_service():
    """Mock the LLM service for shade merging."""
    service = MagicMock()
    # Mock the call_with_retry method for merging
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content='''
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
            )
        )
    ]
    service.call_with_retry.return_value = response
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
            confidence=0.8,
            desc_third_view="Description of test shade 1",
            content_third_view="Detail about test shade 1",
            metadata={}
        ),
        L1Shade(
            id="shade2",
            name="Test Shade 2",
            user_id="test-user",
            summary="This is test shade 2",
            desc_third_view="Description of test shade 2",
            content_third_view="Detail about test shade 2",
            confidence=0.9,
            metadata={}
        )
    ] 