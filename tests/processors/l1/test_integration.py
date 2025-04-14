import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.processors.l1.l1_manager import L1Manager
from app.processors.l1.l1_generator import L1Generator
from app.processors.l1.topics_generator import TopicsGenerator
from app.processors.l1.shade_generator import ShadeGenerator
from app.processors.l1.biography_generator import BiographyGenerator
from app.models.l1.note import Note, Chunk
from app.models.l1.bio import Bio
from app.providers.l1.postgres_adapter import PostgresAdapter
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter
from app.providers.l1.weaviate_adapter import WeaviateAdapter


@pytest.fixture
def integration_test_notes():
    """Return a set of notes for integration testing."""
    # Create embeddings
    embeddings = [np.random.rand(384).tolist() for _ in range(6)]
    
    # Create chunks
    chunks = [
        [
            Chunk(
                id=f"chunk_{i}_0",
                content=f"This is test chunk 0 for document {i}.",
                embedding=embeddings[i*2],
                document_id=f"doc_{i}"
            ),
            Chunk(
                id=f"chunk_{i}_1",
                content=f"This is test chunk 1 for document {i}.",
                embedding=embeddings[i*2 + 1],
                document_id=f"doc_{i}"
            )
        ]
        for i in range(3)
    ]
    
    # Create notes
    notes = [
        Note(
            id=f"doc_{i}",
            content=f"This is test document {i} with multiple paragraphs of content. "
                    f"It contains information about topic {i}. "
                    f"This document has been created for testing the L1 layer processing.",
            create_time="2023-01-0{i}T00:00:00".format(i=i+1),
            embedding=embeddings[i*2],  # Use first chunk embedding for document
            chunks=chunks[i],
            title=f"Test Document {i}",
            summary={"text": f"Summary of document {i}"},
            insight={"text": f"Insight from document {i}"},
            tags=[f"tag{i}", "test"],
            memory_type="TEXT"
        )
        for i in range(3)
    ]
    
    return notes


@pytest.fixture
def mock_llm_service_for_integration():
    """Return a mock LLM service for integration testing."""
    mock = MagicMock()
    
    # Set up return values for different LLM calls in the integration flow
    
    # For topics generation
    topics_response = {
        "choices": [
            {
                "message": {
                    "content": '{"topic": "Test Topic", "tags": ["test", "integration", "example"]}'
                }
            }
        ]
    }
    
    # For shade generation
    shade_response = {
        "choices": [
            {
                "message": {
                    "content": '{"name": "Integration Test Shade", "summary": "This is a test shade for integration testing.", "confidence": 0.8}'
                }
            }
        ]
    }
    
    # For shade merging
    merged_response = {
        "choices": [
            {
                "message": {
                    "content": '[{"name": "Merged Integration Shade", "summary": "This is a merged shade for integration testing.", "confidence": 0.85}]'
                }
            }
        ]
    }
    
    # For biography generation
    bio_response = {
        "choices": [
            {
                "message": {
                    "content": '{"content_third_view": "This is a third-person biography for integration testing.", "summary_third_view": "Integration test bio.", "confidence": 0.9}'
                }
            }
        ]
    }
    
    # For perspective shifting
    perspective_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a first-person perspective shift."
                }
            }
        ]
    }
    
    # Set up mock to return appropriate response based on prompt context
    def mock_completion(messages, **kwargs):
        # Check message content to determine which response to return
        system_content = messages[0].get("content", "").lower() if messages else ""
        user_content = messages[1].get("content", "").lower() if len(messages) > 1 else ""
        
        if "topic" in system_content and "tags" in user_content:
            return topics_response
        elif "shade" in system_content and "documents" in user_content:
            return shade_response
        elif "merge" in system_content and "shades" in user_content:
            return merged_response
        elif "biograph" in system_content:
            return bio_response
        elif "perspective" in system_content:
            return perspective_response
        else:
            # Default response
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Default integration test response"
                        }
                    }
                ]
            }
    
    mock.chat_completion.side_effect = mock_completion
    mock.generate_embeddings.return_value = [np.random.rand(384).tolist() for _ in range(10)]
    
    return mock


@pytest.fixture
def mock_adapters_for_integration():
    """Return mock adapters for integration testing."""
    postgres_mock = MagicMock(spec=PostgresAdapter)
    wasabi_mock = MagicMock(spec=WasabiStorageAdapter)
    weaviate_mock = MagicMock(spec=WeaviateAdapter)
    
    # Configure PostgreSQL adapter
    postgres_mock.get_latest_version.return_value = 1
    postgres_mock.create_version.return_value = MagicMock()
    postgres_mock.update_version_status.return_value = None
    
    # Configure Wasabi adapter
    wasabi_mock.store_json.return_value = None
    
    return postgres_mock, wasabi_mock, weaviate_mock


@patch('app.services.llm_service.LLMService')
def test_topics_to_shades_pipeline(mock_llm_service_cls, mock_llm_service_for_integration, integration_test_notes):
    """Test the pipeline from topics generation to shade generation."""
    # Set up LLM service mock
    mock_llm_service_cls.return_value = mock_llm_service_for_integration
    
    # Create real instances of the generators
    topics_generator = TopicsGenerator(llm_service=mock_llm_service_for_integration)
    shade_generator = ShadeGenerator(
        llm_service=mock_llm_service_for_integration,
        wasabi_adapter=MagicMock(spec=WasabiStorageAdapter)
    )
    
    # Test the TopicsGenerator
    topics_result = topics_generator.generate_topics(integration_test_notes)
    
    # Check that topics were generated
    assert topics_result is not None
    assert len(topics_result) > 0
    
    # Extract a cluster
    cluster_id, cluster = next(iter(topics_result.items()))
    assert "docIds" in cluster
    assert "contents" in cluster
    assert "topic" in cluster
    
    # Now, create cluster notes for the first cluster
    cluster_notes = [
        note for note in integration_test_notes 
        if note.id in cluster["docIds"]
    ]
    
    # Test the ShadeGenerator with the cluster notes
    shade = shade_generator.generate_shade_for_cluster(
        user_id="test_user",
        old_shades=[],
        cluster_notes=cluster_notes,
        memory_list=[]
    )
    
    # Check that a shade was generated
    assert shade is not None
    assert shade.name == "Integration Test Shade"
    assert shade.summary == "This is a test shade for integration testing."
    assert shade.confidence == 0.8


@patch('app.services.llm_service.LLMService')
def test_shade_to_bio_pipeline(mock_llm_service_cls, mock_llm_service_for_integration, integration_test_notes):
    """Test the pipeline from shade generation to biography generation."""
    # Set up LLM service mock
    mock_llm_service_cls.return_value = mock_llm_service_for_integration
    
    # Create real instances of the generators with mock dependencies
    shade_generator = ShadeGenerator(
        llm_service=mock_llm_service_for_integration,
        wasabi_adapter=MagicMock(spec=WasabiStorageAdapter)
    )
    
    biography_generator = BiographyGenerator(
        llm_service=mock_llm_service_for_integration,
        wasabi_adapter=MagicMock(spec=WasabiStorageAdapter)
    )
    
    # Create test shades
    shades = []
    for i in range(2):
        shade = shade_generator.generate_shade_for_cluster(
            user_id="test_user",
            old_shades=[],
            cluster_notes=integration_test_notes[i:i+1],
            memory_list=[]
        )
        if shade:
            shades.append(shade)
    
    # Merge shades
    merged_shades = shade_generator.merge_shades("test_user", shades)
    
    # Check merged shades
    assert merged_shades is not None
    assert len(merged_shades) > 0
    assert merged_shades[0]["name"] == "Merged Integration Shade"
    
    # Create a sample Bio with the merged shades
    sample_bio = Bio(
        shades_list=merged_shades
    )
    
    # Test biography generation
    bio = biography_generator.generate_global_biography(
        user_id="test_user",
        old_profile=sample_bio,
        cluster_list=[{"topic": "Test Topic"}]
    )
    
    # Check that a biography was generated
    assert bio is not None
    assert bio.content_third_view == "This is a third-person biography for integration testing."
    assert bio.summary_third_view == "Integration test bio."
    assert bio.content_first_view == "This is a first-person perspective shift."


@patch('app.services.llm_service.LLMService')
@patch('app.processors.l1.l1_manager.L1Manager._extract_notes_from_l0')
def test_end_to_end_l1_generation(mock_extract, mock_llm_service_cls, mock_llm_service_for_integration, 
                                   mock_adapters_for_integration, integration_test_notes):
    """Test end-to-end L1 generation process."""
    # Set up LLM service mock
    mock_llm_service_cls.return_value = mock_llm_service_for_integration
    
    # Set up extraction mock
    mock_extract.return_value = (
        integration_test_notes, 
        [{"memoryId": note.id, "embedding": note.embedding} for note in integration_test_notes]
    )
    
    # Create adapters
    postgres_mock, wasabi_mock, weaviate_mock = mock_adapters_for_integration
    
    # Create real instances of all components
    topics_generator = TopicsGenerator(llm_service=mock_llm_service_for_integration)
    shade_generator = ShadeGenerator(
        llm_service=mock_llm_service_for_integration,
        wasabi_adapter=wasabi_mock
    )
    biography_generator = BiographyGenerator(
        llm_service=mock_llm_service_for_integration,
        wasabi_adapter=wasabi_mock
    )
    
    l1_generator = L1Generator(
        topics_generator=topics_generator,
        shade_generator=shade_generator,
        biography_generator=biography_generator
    )
    
    l1_manager = L1Manager(
        postgres_adapter=postgres_mock,
        wasabi_adapter=wasabi_mock,
        weaviate_adapter=weaviate_mock,
        l1_generator=l1_generator
    )
    
    # Run the end-to-end process
    result = l1_manager.generate_l1_from_l0("test_user")
    
    # Check the result
    assert result is not None
    assert result.status == "completed"
    assert result.bio is not None
    assert result.clusters is not None
    
    # Verify that adapters were called for storage
    postgres_mock.update_version_status.assert_called_with(
        "test_user", 2, "completed"
    ) 