import pytest
from unittest.mock import MagicMock
from app.models.l1.bio import Bio
from app.models.l1.shade import Shade as L1Shade


@pytest.fixture
def mock_llm_service():
    """Return a mock LLM service that returns predictable responses."""
    mock_service = MagicMock()
    
    # Configure the chat_completion method to return different responses
    # based on the system prompt in the request
    def mock_completion(messages):
        system_message = messages[0]['content'] if messages and messages[0]['role'] == 'system' else ''
        
        # For biography generation (matches both old and new prompts)
        if "creating detailed, insightful, and accurate biographies" in system_message or "clever and perceptive individual" in system_message:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"content_third_view": "They are a test user.", "summary_third_view": "Test user.", "confidence": 0.9}'
                        }
                    }
                ]
            }
        # For perspective shifting
        elif "transforming narratives between different grammatical perspectives" in system_message:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "I am a test user."
                        }
                    }
                ]
            }
        # For status biography
        elif "creating concise status biographies" in system_message or "analyzing and organizing user's memory" in system_message:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"content_third_view": "They are a test user.", "summary_third_view": "Test user.", "health_status": "They appear to be in good health."}'
                        }
                    }
                ]
            }
        # For shade generation (test_shade_generator.py)
        elif "extracting coherent knowledge aspects" in system_message:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"name": "Test Shade", "summary": "This is a test shade summary", "confidence": 0.85}'
                        }
                    }
                ]
            }
        # For shade merging
        elif "merging multiple related knowledge aspects" in system_message:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"name": "Merged Shade", "summary": "This is a merged shade summary", "confidence": 0.9}]'
                        }
                    }
                ]
            }
        # Default response
        else:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Default mock response"
                        }
                    }
                ]
            }
    
    mock_service.chat_completion.side_effect = mock_completion
    
    # For direct text generation
    mock_service.generate_text.return_value = "I am a test user."
    
    return mock_service


@pytest.fixture
def mock_wasabi_adapter():
    """Return a mock Wasabi adapter that tracks storage operations."""
    mock_adapter = MagicMock()
    
    # Configure the store_json method to return a path
    mock_adapter.store_json.return_value = "path/to/test/file.json"
    
    # Configure the retrieve_json method to return test data
    mock_adapter.retrieve_json.return_value = {"test": "data"}
    
    return mock_adapter


@pytest.fixture
def sample_bio():
    """Return a sample Bio instance for testing."""
    return Bio(
        content_first_view="I am a test user.",
        summary_first_view="I'm testing.",
        content_second_view="You are a test user.",
        summary_second_view="You're testing.",
        content_third_view="They are a test user.",
        summary_third_view="They're testing.",
        health_status="They appear to be in good health and focused on their testing tasks.",
        shades_list=[
            {"name": "Test Shade 1", "summary": "Summary of test shade 1"},
            {"name": "Test Shade 2", "summary": "Summary of test shade 2"}
        ]
    )


@pytest.fixture
def sample_clusters():
    """Return sample clusters data for testing."""
    return {
        "clusterList": [
            {
                "topic": "Test Topic 1", 
                "tags": ["test", "sample"], 
                "documents": ["doc1", "doc2"]
            },
            {
                "topic": "Test Topic 2", 
                "tags": ["example", "demo"], 
                "documents": ["doc3", "doc4"]
            }
        ]
    }


@pytest.fixture
def sample_notes():
    """Return sample notes for testing."""
    from app.models.l1.note import Note
    
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
            embedding=[0.4, 0.5, 0.6]
        )
    ]


@pytest.fixture
def sample_shades():
    """Return sample shades for testing."""
    return [
        L1Shade(
            id="shade1",
            name="Test Shade 1",
            summary="This is test shade 1",
            content="Detail about test shade 1",
            confidence=0.8,
            source_clusters=["cluster1"]
        ),
        L1Shade(
            id="shade2",
            name="Test Shade 2",
            summary="This is test shade 2",
            content="Detail about test shade 2",
            confidence=0.7,
            source_clusters=["cluster2"]
        )
    ] 