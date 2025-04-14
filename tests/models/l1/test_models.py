import unittest
from datetime import datetime
from app.models.l1 import (
    Note, Chunk, Topic, Cluster, Memory, 
    Shade, ShadeInfo, ShadeMergeInfo, MergedShadeResult,
    Bio, L1GenerationResult
)


class TestL1Models(unittest.TestCase):
    """Test cases for L1 data models."""
    
    def test_chunk(self):
        """Test Chunk model."""
        # Create a chunk
        chunk = Chunk(
            id="chunk1",
            content="This is a test chunk",
            embedding=[0.1, 0.2, 0.3],
            document_id="doc1",
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        chunk_dict = chunk.to_dict()
        chunk2 = Chunk.from_dict(chunk_dict)
        
        self.assertEqual(chunk.id, chunk2.id)
        self.assertEqual(chunk.content, chunk2.content)
        self.assertEqual(chunk.embedding, chunk2.embedding)
        self.assertEqual(chunk.document_id, chunk2.document_id)
        self.assertEqual(chunk.metadata, chunk2.metadata)
    
    def test_note(self):
        """Test Note model."""
        # Create chunks
        chunk1 = Chunk(id="chunk1", content="Chunk 1", embedding=[0.1, 0.2, 0.3])
        chunk2 = Chunk(id="chunk2", content="Chunk 2", embedding=[0.4, 0.5, 0.6])
        
        # Create a note
        note = Note(
            id="note1",
            content="This is a test note",
            create_time=datetime.now(),
            embedding=[0.1, 0.2, 0.3],
            chunks=[chunk1, chunk2],
            title="Test Note",
            summary={"summary": "This is a summary"},
            insight={"insight": "This is an insight"},
            tags=["test", "note"],
            memory_type="TEXT",
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        note_dict = note.to_dict()
        note2 = Note.from_dict(note_dict)
        
        self.assertEqual(note.id, note2.id)
        self.assertEqual(note.content, note2.content)
        self.assertEqual(note.title, note2.title)
        self.assertEqual(len(note.chunks), len(note2.chunks))
        self.assertEqual(note.chunks[0].id, note2.chunks[0].id)
    
    def test_topic(self):
        """Test Topic model."""
        # Create a topic
        topic = Topic(
            id="topic1",
            name="Test Topic",
            summary="This is a test topic",
            document_ids=["doc1", "doc2"],
            embedding=[0.1, 0.2, 0.3]
        )
        
        # Test serialization/deserialization
        topic_dict = topic.to_dict()
        topic2 = Topic.from_dict(topic_dict)
        
        self.assertEqual(topic.id, topic2.id)
        self.assertEqual(topic.name, topic2.name)
        self.assertEqual(topic.summary, topic2.summary)
        self.assertEqual(topic.document_ids, topic2.document_ids)
        self.assertEqual(topic.embedding, topic2.embedding)
    
    def test_memory(self):
        """Test Memory model."""
        # Create a memory
        memory = Memory(
            memory_id="mem1",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        memory_dict = memory.to_dict()
        memory2 = Memory.from_dict(memory_dict)
        
        self.assertEqual(memory.memory_id, memory2.memory_id)
        self.assertEqual(memory.embedding, memory2.embedding)
        self.assertEqual(memory.metadata, memory2.metadata)
    
    def test_cluster(self):
        """Test Cluster model."""
        # Create memories
        memory1 = Memory(memory_id="mem1", embedding=[0.1, 0.2, 0.3])
        memory2 = Memory(memory_id="mem2", embedding=[0.4, 0.5, 0.6])
        
        # Create a cluster
        cluster = Cluster(
            id="cluster1",
            topic_id="topic1",
            name="Test Cluster",
            summary="This is a test cluster",
            memory_list=[memory1, memory2],
            center_embedding=[0.25, 0.35, 0.45]
        )
        
        # Test properties
        self.assertEqual(cluster.document_count, 2)
        self.assertEqual(cluster.document_ids, ["mem1", "mem2"])
        
        # Test serialization/deserialization
        cluster_dict = cluster.to_dict()
        cluster2 = Cluster.from_dict(cluster_dict)
        
        self.assertEqual(cluster.id, cluster2.id)
        self.assertEqual(cluster.topic_id, cluster2.topic_id)
        self.assertEqual(cluster.name, cluster2.name)
        self.assertEqual(cluster.summary, cluster2.summary)
        self.assertEqual(len(cluster.memory_list), len(cluster2.memory_list))
        self.assertEqual(cluster.memory_list[0].memory_id, cluster2.memory_list[0].memory_id)
    
    def test_shade_info(self):
        """Test ShadeInfo model."""
        # Create a shade info
        shade_info = ShadeInfo(
            shade_id="shade1",
            name="Test Shade",
            content="This is a test shade",
            confidence=0.85,
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        shade_info_dict = shade_info.to_dict()
        shade_info2 = ShadeInfo.from_dict(shade_info_dict)
        
        self.assertEqual(shade_info.shade_id, shade_info2.shade_id)
        self.assertEqual(shade_info.name, shade_info2.name)
        self.assertEqual(shade_info.content, shade_info2.content)
        self.assertEqual(shade_info.confidence, shade_info2.confidence)
        self.assertEqual(shade_info.metadata, shade_info2.metadata)
    
    def test_shade(self):
        """Test Shade model."""
        # Create a shade
        shade = Shade(
            id="shade1",
            name="Test Shade",
            summary="This is a test shade",
            content="Detailed content of the shade",
            confidence=0.85,
            source_clusters=["cluster1", "cluster2"]
        )
        
        # Test serialization/deserialization
        shade_dict = shade.to_dict()
        shade2 = Shade.from_dict(shade_dict)
        
        self.assertEqual(shade.id, shade2.id)
        self.assertEqual(shade.name, shade2.name)
        self.assertEqual(shade.summary, shade2.summary)
        self.assertEqual(shade.content, shade2.content)
        self.assertEqual(shade.confidence, shade2.confidence)
        self.assertEqual(shade.source_clusters, shade2.source_clusters)
    
    def test_shade_merge_info(self):
        """Test ShadeMergeInfo model."""
        # Create a shade merge info
        shade_merge_info = ShadeMergeInfo(
            shade_id="shade1",
            name="Test Shade",
            summary="This is a test shade",
            content="Detailed content of the shade",
            confidence=0.85,
            source_clusters=["cluster1", "cluster2"],
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        shade_merge_info_dict = shade_merge_info.to_dict()
        shade_merge_info2 = ShadeMergeInfo.from_dict(shade_merge_info_dict)
        
        self.assertEqual(shade_merge_info.shade_id, shade_merge_info2.shade_id)
        self.assertEqual(shade_merge_info.name, shade_merge_info2.name)
        self.assertEqual(shade_merge_info.summary, shade_merge_info2.summary)
        self.assertEqual(shade_merge_info.content, shade_merge_info2.content)
        self.assertEqual(shade_merge_info.confidence, shade_merge_info2.confidence)
        self.assertEqual(shade_merge_info.source_clusters, shade_merge_info2.source_clusters)
        self.assertEqual(shade_merge_info.metadata, shade_merge_info2.metadata)
        
        # Test creation from Shade
        shade = Shade(
            id="shade1",
            name="Test Shade",
            summary="This is a test shade",
            content="Detailed content of the shade",
            confidence=0.85,
            source_clusters=["cluster1", "cluster2"],
            metadata={"source": "test"}
        )
        
        shade_merge_info3 = ShadeMergeInfo.from_shade(shade)
        
        self.assertEqual(shade.id, shade_merge_info3.shade_id)
        self.assertEqual(shade.name, shade_merge_info3.name)
        self.assertEqual(shade.summary, shade_merge_info3.summary)
        self.assertEqual(shade.content, shade_merge_info3.content)
        self.assertEqual(shade.confidence, shade_merge_info3.confidence)
        self.assertEqual(shade.source_clusters, shade_merge_info3.source_clusters)
        self.assertEqual(shade.metadata, shade_merge_info3.metadata)
    
    def test_merged_shade_result(self):
        """Test MergedShadeResult model."""
        # Create a merged shade result
        merged_shade_result = MergedShadeResult(
            success=True,
            merge_shade_list=[
                {"id": "merged1", "name": "Merged Shade 1"},
                {"id": "merged2", "name": "Merged Shade 2"}
            ]
        )
        
        # Test serialization/deserialization
        merged_shade_result_dict = merged_shade_result.to_dict()
        merged_shade_result2 = MergedShadeResult.from_dict(merged_shade_result_dict)
        
        self.assertEqual(merged_shade_result.success, merged_shade_result2.success)
        self.assertEqual(len(merged_shade_result.merge_shade_list), len(merged_shade_result2.merge_shade_list))
        self.assertEqual(merged_shade_result.merge_shade_list[0]["id"], merged_shade_result2.merge_shade_list[0]["id"])
    
    def test_bio(self):
        """Test Bio model."""
        # Create a bio
        bio = Bio(
            content_first_view="I am a user with interests in technology",
            summary_first_view="I like technology",
            content_second_view="You are a user with interests in technology",
            summary_second_view="You like technology",
            content_third_view="They are a user with interests in technology",
            summary_third_view="They like technology",
            confidence=0.9,
            shades_list=[
                {"id": "shade1", "name": "Technology Interest"}
            ]
        )
        
        # Test methods
        str_rep = bio.to_str()
        self.assertIn("They are a user with interests in technology", str_rep)
        
        complete_content = bio.complete_content()
        self.assertTrue(complete_content.startswith("They like technology"))
        self.assertIn("They are a user with interests in technology", complete_content)
        
        # Test serialization/deserialization
        bio_dict = bio.to_dict()
        bio2 = Bio.from_dict(bio_dict)
        
        self.assertEqual(bio.content_first_view, bio2.content_first_view)
        self.assertEqual(bio.summary_first_view, bio2.summary_first_view)
        self.assertEqual(bio.content_second_view, bio2.content_second_view)
        self.assertEqual(bio.summary_second_view, bio2.summary_second_view)
        self.assertEqual(bio.content_third_view, bio2.content_third_view)
        self.assertEqual(bio.summary_third_view, bio2.summary_third_view)
        self.assertEqual(bio.confidence, bio2.confidence)
        self.assertEqual(bio.shades_list, bio2.shades_list)
    
    def test_l1_generation_result(self):
        """Test L1GenerationResult model."""
        # Create a bio
        bio = Bio(
            content_third_view="They are a user with interests in technology",
            summary_third_view="They like technology"
        )
        
        # Create an L1 generation result
        result = L1GenerationResult(
            bio=bio,
            clusters={
                "clusterList": [
                    {"id": "cluster1", "name": "Cluster 1"}
                ],
                "outlierList": []
            },
            chunk_topics=[
                {"id": "topic1", "name": "Topic 1"}
            ]
        )
        
        # Test serialization/deserialization
        result_dict = result.to_dict()
        result2 = L1GenerationResult.from_dict(result_dict)
        
        self.assertEqual(result.bio.content_third_view, result2.bio.content_third_view)
        self.assertEqual(result.clusters["clusterList"][0]["id"], result2.clusters["clusterList"][0]["id"])
        self.assertEqual(result.chunk_topics[0]["id"], result2.chunk_topics[0]["id"])
        self.assertEqual(result.status, result2.status)
        
        # Test factory methods
        success_result = L1GenerationResult.success(bio, result.clusters, result.chunk_topics)
        self.assertEqual(success_result.status, "completed")
        self.assertEqual(success_result.bio, bio)
        
        failure_result = L1GenerationResult.failure("Test error")
        self.assertEqual(failure_result.status, "failed")
        self.assertEqual(failure_result.error, "Test error")


if __name__ == "__main__":
    unittest.main() 