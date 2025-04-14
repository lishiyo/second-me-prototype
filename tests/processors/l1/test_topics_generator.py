import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.processors.l1.topics_generator import TopicsGenerator
from app.models.l1.note import Note, Chunk


@pytest.fixture
def topics_generator(mock_llm_service):
    """Return a TopicsGenerator instance with a mock LLM service."""
    return TopicsGenerator(llm_service=mock_llm_service)


def test_init():
    """Test TopicsGenerator initialization."""
    generator = TopicsGenerator()
    assert generator.default_cophenetic_distance == 0.7
    assert generator.default_outlier_cutoff_distance == 0.9
    assert generator.default_cluster_merge_distance == 0.75


def test_generate_topics_empty_notes(topics_generator):
    """Test generating topics with empty notes list."""
    result = topics_generator.generate_topics([])
    assert result == {} or result is None


def test_generate_topics(topics_generator, sample_notes):
    """Test generating topics from notes."""
    result = topics_generator.generate_topics(sample_notes)
    
    # Check that we have a non-empty result
    assert result
    assert isinstance(result, dict)
    
    # Check that some clusters were created
    assert len(result) > 0
    
    # Check structure of a cluster
    cluster_id, cluster = next(iter(result.items()))
    assert "indices" in cluster
    assert "docIds" in cluster
    assert "contents" in cluster
    assert "embedding" in cluster
    assert "chunkIds" in cluster
    assert "tags" in cluster
    assert "topic" in cluster
    assert "topicId" in cluster


def test_generate_topics_for_shades_empty(topics_generator):
    """Test generating topics for shades with empty input."""
    result = topics_generator.generate_topics_for_shades([], [], [])
    
    # Should return a structure with empty lists
    assert result
    assert "clusterList" in result
    assert "outlierMemoryList" in result
    assert len(result["clusterList"]) == 0
    assert len(result["outlierMemoryList"]) == 0


def test_generate_topics_for_shades_cold_start(topics_generator):
    """Test generating topics for shades with no existing clusters (cold start)."""
    new_memory_list = [
        {"memoryId": f"mem_{i}", "embedding": [0.1 * i, 0.2 * i, 0.3 * i]}
        for i in range(5)
    ]
    
    result = topics_generator.generate_topics_for_shades([], [], new_memory_list)
    
    # Check that clusters were created
    assert result
    assert "clusterList" in result
    assert len(result["clusterList"]) > 0
    
    # Check a cluster structure
    cluster = result["clusterList"][0]
    assert "clusterId" in cluster
    assert "topic" in cluster
    assert "tags" in cluster
    assert "memoryList" in cluster


def test_convert_memories_to_notes(topics_generator):
    """Test converting memory list to notes."""
    memory_list = [
        {
            "memoryId": f"mem_{i}",
            "content": f"Test content {i}",
            "createTime": "2023-01-01T00:00:00",
            "embedding": [0.1, 0.2, 0.3],
            "title": f"Test title {i}",
            "memoryType": "TEXT",
            "chunks": [
                {
                    "id": f"chunk_{i}_0",
                    "content": f"Test chunk content {i}_0",
                    "embedding": [0.1, 0.2, 0.3]
                }
            ]
        }
        for i in range(3)
    ]
    
    notes = topics_generator._convert_memories_to_notes(memory_list)
    
    # Check conversion
    assert len(notes) == 3
    assert all(isinstance(note, Note) for note in notes)
    assert notes[0].id == "mem_0"
    assert notes[0].content == "Test content 0"
    assert len(notes[0].chunks) == 1
    assert notes[0].chunks[0].id == "chunk_0_0"


def test_cold_start(topics_generator, sample_notes):
    """Test cold start clustering."""
    result = topics_generator._cold_start(sample_notes)
    
    # Check result structure
    assert result
    assert isinstance(result, dict)
    assert len(result) > 0


def test_build_embedding_chunks(topics_generator, sample_notes):
    """Test building embedding matrix and clean chunks."""
    embedding_matrix, clean_chunks, all_note_ids = topics_generator._build_embedding_chunks(sample_notes)
    
    # Check results
    assert len(embedding_matrix) > 0
    assert len(clean_chunks) > 0
    assert len(all_note_ids) == len(sample_notes)
    assert all(isinstance(e, list) for e in embedding_matrix)
    assert all(isinstance(c, Chunk) for c in clean_chunks)


def test_hierarchical_clustering(topics_generator):
    """Test hierarchical clustering."""
    # Create a simple embedding matrix
    embedding_matrix = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],  # Close to first point
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],  # Close to third point
    ]
    
    clusters = topics_generator._hierarchical_clustering(embedding_matrix, 0.2)
    
    # With a small distance threshold, we should get multiple clusters
    assert len(clusters) > 1
    # The first two points should be in one cluster, and the second two in another
    cluster_ids = list(clusters.keys())
    assert len(clusters[cluster_ids[0]]) + len(clusters[cluster_ids[1]]) == 4


def test_generate_topic_from_chunks(topics_generator, sample_chunks):
    """Test generating topics for chunks."""
    # Take a subset to speed up test
    chunks_subset = sample_chunks[:2]
    
    chunks_with_topics = topics_generator._generate_topic_from_chunks(chunks_subset)
    
    # Check that topics were generated
    assert len(chunks_with_topics) == len(chunks_subset)
    assert all(hasattr(chunk, 'topic') for chunk in chunks_with_topics)
    assert all(hasattr(chunk, 'tags') for chunk in chunks_with_topics)


def test_gen_cluster_data(topics_generator, sample_chunks):
    """Test generating cluster data."""
    # Generate topics for the chunks first
    chunks_with_topics = []
    for chunk in sample_chunks[:2]:
        chunk.topic = "Test Topic"
        chunk.tags = ["test", "topic"]
        chunks_with_topics.append(chunk)
    
    # Create test clusters
    clusters = {
        "0": [0],
        "1": [1]
    }
    
    cluster_data = topics_generator._gen_cluster_data(clusters, chunks_with_topics)
    
    # Check results
    assert len(cluster_data) == 2
    assert "0" in cluster_data
    assert "1" in cluster_data
    assert "indices" in cluster_data["0"]
    assert "docIds" in cluster_data["0"]
    assert "topic" in cluster_data["0"]
    assert "tags" in cluster_data["0"]


def test_gen_cluster_topic(topics_generator):
    """Test generating a combined topic and tags for a cluster."""
    c_tags = [["test", "topic1"], ["example", "topic2"]]
    c_topics = ["Test Topic 1", "Test Topic 2"]
    
    new_tags, new_topic = topics_generator._gen_cluster_topic(c_tags, c_topics)
    
    # Check results
    assert new_tags
    assert new_topic
    assert isinstance(new_tags, list)
    assert isinstance(new_topic, str)


def test_parse_response(topics_generator):
    """Test parsing JSON response."""
    content = '{"topic": "Test Topic", "tags": ["test", "topic", "example"]}'
    
    topic, tags = topics_generator._parse_response(content, "topic", "tags")
    
    # Check parsing
    assert topic == "Test Topic"
    assert tags == ["test", "topic", "example"]
    
    # Test with malformed JSON
    malformed = "Not a JSON {missing: quotes}"
    topic, tags = topics_generator._parse_response(malformed, "topic", "tags")
    assert topic == "" or topic is None
    assert tags == [] or tags is None 