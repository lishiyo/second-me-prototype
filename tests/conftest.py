import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import json

import numpy as np

from app.models.l1.note import Note, Chunk
from app.models.l1.bio import Bio
from app.models.l1.db_models import L1Shade
from app.providers.l1.postgres_adapter import PostgresAdapter
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter
from app.providers.l1.weaviate_adapter import WeaviateAdapter
from app.services.llm_service import LLMService


@pytest.fixture
def sample_embeddings():
    """Return sample embeddings for testing."""
    # Create 5 random 384-dimensional embeddings
    return [list(np.random.rand(384)) for _ in range(5)]


@pytest.fixture
def sample_chunks(sample_embeddings):
    """Return sample chunks for testing."""
    return [
        Chunk(
            id=f"chunk_{i}",
            content=f"This is test chunk content {i}",
            embedding=sample_embeddings[i],
            document_id=f"doc_{i//2}"
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_notes(sample_chunks, sample_embeddings):
    """Return sample notes for testing."""
    return [
        Note(
            id=f"doc_{i}",
            content=f"This is test document content {i}. It contains various information about topic {i}.",
            create_time=datetime.now(),
            embedding=sample_embeddings[i],
            chunks=[chunk for chunk in sample_chunks if chunk.document_id == f"doc_{i}"],
            title=f"Test Document {i}",
            summary={"text": f"Summary of document {i}"},
            insight={"text": f"Insight from document {i}"},
            tags=[f"tag{i}", "test", f"topic{i}"],
            memory_type="TEXT"
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_clusters():
    """Return sample cluster data for testing."""
    return {
        "clusterList": [
            {
                "clusterId": "cluster_0",
                "topic": "Test Topic 0",
                "tags": ["test", "topic0", "sample"],
                "memoryList": [
                    {"memoryId": "doc_0", "embedding": [0.1, 0.2, 0.3]}
                ]
            },
            {
                "clusterId": "cluster_1",
                "topic": "Test Topic 1",
                "tags": ["test", "topic1", "example"],
                "memoryList": [
                    {"memoryId": "doc_1", "embedding": [0.2, 0.3, 0.4]},
                    {"memoryId": "doc_2", "embedding": [0.3, 0.4, 0.5]}
                ]
            }
        ],
        "outlierMemoryList": []
    }


@pytest.fixture
def sample_shades():
    """Return sample shades for testing."""
    return [
        L1Shade(
            id=f"shade_{i}",
            user_id="test_user",
            name=f"Test Shade {i}",
            summary=f"This is a summary of test shade {i}",
            confidence=0.8,
            s3_path=f"l1/shades/test_user/shade_{i}.json"
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_bio():
    """Return a sample bio for testing."""
    return Bio(
        content_first_view="I am a test user with interests in technology.",
        summary_first_view="I like technology.",
        content_second_view="You are a test user with interests in technology.",
        summary_second_view="You like technology.",
        content_third_view="They are a test user with interests in technology.",
        summary_third_view="They like technology.",
        confidence=0.9,
        shades_list=[{"id": f"shade_{i}", "name": f"Test Shade {i}"} for i in range(2)]
    )


@pytest.fixture
def mock_llm_service():
    """Return a mock LLM service that provides deterministic responses for testing."""
    mock_service = MagicMock(spec=LLMService)
    
    # Configure the chat_completion method to return different outputs
    # based on the prompt content
    def mock_chat_completion(messages, model=None, temperature=None, max_tokens=None, 
                           top_p=None, frequency_penalty=None, presence_penalty=None, 
                           seed=None, response_format=None, timeout=None, **kwargs):
        """Mock chat completion that returns different responses based on prompt content."""
        system_content = ''
        user_content = ''
        
        # Extract system and user content from messages
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
        
        # Create a response class with the expected attributes
        class MockResponse:
            def __init__(self, content):
                self.choices = [MagicMock()]
                self.choices[0].message = MagicMock()
                self.choices[0].message.content = content
        
        # Generate different responses based on the content
        if 'clever and perceptive individual' in system_content.lower():
            # Global biography response (SYS_BIO)
            content = json.dumps({
                "content_third_view": "They are a test user with various interests.",
                "summary_third_view": "They are a test user.",
                "confidence": 0.85
            })
        elif 'analyzing and organizing user\'s memory' in system_content.lower():
            # Status biography response (SYS_STATUS)
            content = json.dumps({
                "content_third_view": "They are a test user with several recent activities.",
                "summary_third_view": "Test user with recent activities.",
                "health_status": "They appear to be in good health."
            })
        elif 'expert system for transforming narratives' in system_content.lower():
            # Perspective shifting (SYS_PERSPECTIVE)
            if 'second-person' in user_content.lower():
                content = "You are a test user with various interests."
            elif 'first-person' in user_content.lower():
                content = "I am a test user with various interests."
            else:
                content = "They are a test user with various interests."
        elif 'perspective' in system_content.lower():
            # Common perspective shift system prompt
            content = "You are a test user with various interests."
        elif 'data analysis with psychology' in system_content.lower():
            # For shade generation
            content = json.dumps({
                "name": "Test Shade",
                "summary": "A test shade for unit testing",
                "confidence": 0.85,
                "icon": "🧪",
                "aspect": "Testing",
                "content": "This is a detailed description of the test shade.",
                "desc_second_view": "You are a tester with interests in QA.",
                "desc_third_view": "They are a tester with interests in QA.",
                "content_second_view": "You enjoy testing software and finding bugs.",
                "content_third_view": "They enjoy testing software and finding bugs.",
                "timelines": [
                    {"createTime": "2023-05-15", "description": "Ran unit tests", "refId": "doc1"}
                ]
            })
        else:
            # Default response
            content = "This is a test response from the mock LLM service."
        
        return MockResponse(content)
    
    # Assign the mock function to the chat_completion method
    mock_service.chat_completion.side_effect = mock_chat_completion
    
    return mock_service


@pytest.fixture
def mock_postgres_adapter():
    """Return a mock PostgreSQL adapter."""
    mock_adapter = MagicMock(spec=PostgresAdapter)
    
    # Configure common method responses
    mock_adapter.get_latest_version.return_value = 1
    mock_adapter.create_version.return_value = MagicMock()
    mock_adapter.update_version_status.return_value = None
    
    return mock_adapter


@pytest.fixture
def mock_wasabi_adapter():
    """Return a mock Wasabi adapter."""
    mock_adapter = MagicMock(spec=WasabiStorageAdapter)
    
    # Configure common method responses
    mock_adapter.store_json.return_value = None
    mock_adapter.get_json.return_value = {"data": "test_data"}
    
    return mock_adapter


@pytest.fixture
def mock_weaviate_adapter():
    """Return a mock Weaviate adapter."""
    mock_adapter = MagicMock(spec=WeaviateAdapter)
    
    # Configure common method responses
    mock_adapter.store_embedding.return_value = None
    mock_adapter.get_embedding.return_value = [0.1, 0.2, 0.3]
    
    return mock_adapter 