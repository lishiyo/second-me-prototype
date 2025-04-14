import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

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
    """Return a mock LLM service with predefined responses."""
    mock_service = MagicMock(spec=LLMService)
    
    # Mock responses for different prompts
    topic_response = {
        "choices": [
            {
                "message": {
                    "content": '{"topic": "Test Topic", "tags": ["test", "topic", "example"]}'
                }
            }
        ]
    }
    
    shade_response = {
        "choices": [
            {
                "message": {
                    "content": '{"name": "Test Shade", "summary": "This is a test shade summary", "confidence": 0.85}'
                }
            }
        ]
    }
    
    merged_shades_response = {
        "choices": [
            {
                "message": {
                    "content": '[{"name": "Merged Shade", "summary": "This is a merged shade summary", "confidence": 0.9}]'
                }
            }
        ]
    }
    
    bio_response = {
        "choices": [
            {
                "message": {
                    "content": '{"content_third_view": "They are a test user.", "summary_third_view": "Test user.", "confidence": 0.9}'
                }
            }
        ]
    }
    
    perspective_response = {
        "choices": [
            {
                "message": {
                    "content": "I am a test user."
                }
            }
        ]
    }
    
    # Configure the mock to return different responses based on the prompt
    def mock_chat_completion(messages, **kwargs):
        system_content = messages[0]["content"] if len(messages) > 0 and "content" in messages[0] else ""
        user_content = messages[1]["content"] if len(messages) > 1 and "content" in messages[1] else ""
        
        # For test_gen_cluster_topic - look for specific test values in user_content
        if '"Test Topic 1"' in user_content and '"Test Topic 2"' in user_content:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"topic": "Combined Test Topic", "tags": ["combined", "test", "topic"]}'
                        }
                    }
                ]
            }
        
        # This is a catch-all for any _gen_cluster_topic call
        elif user_content.startswith("Please generate the new topic and new tags"):
            print("MATCHED: user_content starts with 'Please generate the new topic and new tags'")
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"topic": "Combined Test Topic", "tags": ["combined", "test", "topic"]}'
                        }
                    }
                ]
            }
        
        # Specific check for gen_cluster_topic function (directly match the strings from SYS_COMB and USR_COMB)
        elif (system_content.startswith("You are a skilled wordsmith with extensive experience") and 
            "crafting a new topic and a new set of tags" in system_content):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"topic": "Combined Test Topic", "tags": ["combined", "test", "topic"]}'
                        }
                    }
                ]
            }
        # Other conditions (keep these as fallbacks)
        elif system_content.startswith("You are a skilled wordsmith") and "knowledge chunk" in system_content:
            # This is for SYS_TOPICS prompt
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"topic": "Test Topic", "tags": ["test", "topic", "example"]}'
                        }
                    }
                ]
            }
        elif "Topics: " in user_content and "Tags list: " in user_content:
            # This matches the USR_COMB format
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"topic": "Combined Test Topic", "tags": ["combined", "test", "topic"]}'
                        }
                    }
                ]
            }
        elif "topic" in system_content.lower() and "tags" in user_content.lower():
            return topic_response
        elif "shade" in system_content.lower() and "documents" in user_content.lower():
            return shade_response
        elif "merge" in system_content.lower() and "shades" in user_content.lower():
            return merged_shades_response
        elif "biograph" in system_content.lower():
            return bio_response
        elif "perspective" in system_content.lower():
            return perspective_response
        else:
            # For debugging purposes, print what we're receiving
            print(f"UNMATCHED PROMPT - System: {system_content[:50]}... User: {user_content[:50]}...")
            
            # Default response
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"topic": "Default Topic", "tags": ["default", "tags"]}'
                        }
                    }
                ]
            }
    
    mock_service.chat_completion.side_effect = mock_chat_completion
    
    # Mock for embedding generation
    mock_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3] for _ in range(10)]
    
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