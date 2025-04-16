import unittest
import numpy as np
from datetime import datetime
from app.models.l1 import (
    Note, Chunk, Topic, Cluster, Memory, 
    L1Shade, ShadeInfo, ShadeMergeInfo, MergedShadeResult,
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
        self.assertTrue(np.array_equal(chunk.embedding, chunk2.embedding))
        self.assertEqual(chunk.document_id, chunk2.document_id)
        chunk2.metadata = {"source": "test"}
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
        self.assertTrue(np.array_equal(topic.embedding, topic2.embedding))
    
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
        self.assertTrue(np.array_equal(memory.embedding, memory2.embedding))
        expected_metadata = {"source": "test"}
        self.assertEqual(expected_metadata, memory.metadata)
    
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
            metadata={"source": "test"},
            aspect="Personality",
            icon="🧠",
            desc_third_view="Description in third view",
            content_third_view="Content in third view",
            desc_second_view="Description in second view",
            content_second_view="Content in second view"
        )
        
        # Test serialization/deserialization
        shade_info_dict = shade_info.to_dict()
        shade_info2 = ShadeInfo.from_dict(shade_info_dict)
        
        self.assertEqual(shade_info.shade_id, shade_info2.shade_id)
        self.assertEqual(shade_info.name, shade_info2.name)
        self.assertEqual(shade_info.content, shade_info2.content)
        self.assertEqual(shade_info.confidence, shade_info2.confidence)
        self.assertEqual(shade_info.metadata, shade_info2.metadata)
        self.assertEqual(shade_info.aspect, shade_info2.aspect)
        self.assertEqual(shade_info.icon, shade_info2.icon)
        self.assertEqual(shade_info.desc_third_view, shade_info2.desc_third_view)
        self.assertEqual(shade_info.content_third_view, shade_info2.content_third_view)
        self.assertEqual(shade_info.desc_second_view, shade_info2.desc_second_view)
        self.assertEqual(shade_info.content_second_view, shade_info2.content_second_view)
    
    def test_shade(self):
        """Test L1Shade model."""
        # Create a shade
        shade = L1Shade(
            id="shade1",
            name="Test Shade",
            summary="This is a test shade",
            aspect="Personality",
            icon="🧠",
            desc_third_view="Description in third view",
            content_third_view="Content in third view",
            desc_second_view="Description in second view",
            content_second_view="Content in second view",
            confidence=0.85,
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        shade_dict = shade.to_dict()
        shade2 = L1Shade.from_dict(shade_dict)
        
        self.assertEqual(shade.id, shade2.id)
        self.assertEqual(shade.name, shade2.name)
        self.assertEqual(shade.summary, shade2.summary)
        self.assertEqual(shade.aspect, shade2.aspect)
        self.assertEqual(shade.icon, shade2.icon)
        self.assertEqual(shade.desc_third_view, shade2.desc_third_view)
        self.assertEqual(shade.content_third_view, shade2.content_third_view)
        self.assertEqual(shade.desc_second_view, shade2.desc_second_view)
        self.assertEqual(shade.content_second_view, shade2.content_second_view)
        self.assertEqual(shade.confidence, shade2.confidence)
        self.assertEqual(shade.metadata, shade2.metadata)
    
    def test_shade_merge_info(self):
        """Test ShadeMergeInfo model."""
        # Create a shade merge info
        shade_merge_info = ShadeMergeInfo(
            shade_id="shade1",
            name="Test Shade",
            aspect="Personality",
            icon="🧠",
            desc_third_view="This is a description in third person",
            content_third_view="This is content in third person",
            desc_second_view="This is a description in second person",
            content_second_view="This is content in second person",
            cluster_info={"cluster_id": "cluster1", "memory_count": 5},
            metadata={"source": "test"}
        )
        
        # Test serialization/deserialization
        shade_merge_info_dict = shade_merge_info.to_dict()
        shade_merge_info2 = ShadeMergeInfo.from_dict(shade_merge_info_dict)
        
        self.assertEqual(shade_merge_info.shade_id, shade_merge_info2.shade_id)
        self.assertEqual(shade_merge_info.name, shade_merge_info2.name)
        self.assertEqual(shade_merge_info.aspect, shade_merge_info2.aspect)
        self.assertEqual(shade_merge_info.icon, shade_merge_info2.icon)
        self.assertEqual(shade_merge_info.desc_third_view, shade_merge_info2.desc_third_view)
        self.assertEqual(shade_merge_info.content_third_view, shade_merge_info2.content_third_view)
        self.assertEqual(shade_merge_info.desc_second_view, shade_merge_info2.desc_second_view)
        self.assertEqual(shade_merge_info.content_second_view, shade_merge_info2.content_second_view)
        self.assertEqual(shade_merge_info.cluster_info, shade_merge_info2.cluster_info)
        self.assertEqual(shade_merge_info.metadata, shade_merge_info2.metadata)
        
        # Test creation from L1Shade
        shade = L1Shade(
            id="shade1",
            name="Test Shade",
            aspect="Personality",
            icon="🧠",
            desc_third_view="This is a description in third person",
            content_third_view="This is content in third person",
            desc_second_view="This is a description in second person",
            content_second_view="This is content in second person",
            metadata={"source": "test"}
        )
        
        shade_merge_info3 = ShadeMergeInfo.from_shade(shade)
        
        self.assertEqual(shade.id, shade_merge_info3.shade_id)
        self.assertEqual(shade.name, shade_merge_info3.name)
        self.assertEqual(shade.aspect, shade_merge_info3.aspect)
        self.assertEqual(shade.icon, shade_merge_info3.icon)
        self.assertEqual(shade.desc_third_view, shade_merge_info3.desc_third_view)
        self.assertEqual(shade.content_third_view, shade_merge_info3.content_third_view)
        self.assertEqual(shade.desc_second_view, shade_merge_info3.desc_second_view)
        self.assertEqual(shade.content_second_view, shade_merge_info3.content_second_view)
        self.assertEqual(shade.metadata, shade_merge_info3.metadata)
        
        # Test helper methods
        test_desc = "Improved description"
        test_content = "Improved content"
        shade_merge_info.improve_shade_info(test_desc, test_content)
        self.assertEqual(shade_merge_info.desc_third_view, test_desc)
        self.assertEqual(shade_merge_info.content_third_view, test_content)
        
        # Test string representation
        str_rep = shade_merge_info.to_str()
        self.assertIn("**[Name]**: Test Shade", str_rep)
        self.assertIn("**[Aspect]**: Personality", str_rep)
        self.assertIn("**[Icon]**: 🧠", str_rep)
        self.assertIn(test_desc, str_rep)
        self.assertIn(test_content, str_rep)
    
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
            chunk_topics={
                "topic1": {"indices": [0, 1], "docIds": ["doc1", "doc2"], "name": "Topic 1"}
            },
            generate_time=datetime.now()  # Use generate_time instead of generated_at
        )
        
        # Test serialization/deserialization
        result_dict = result.to_dict()
        result2 = L1GenerationResult.from_dict(result_dict)
        
        self.assertEqual(result.bio.content_third_view, result2.bio.content_third_view)
        self.assertEqual(result.clusters["clusterList"][0]["id"], result2.clusters["clusterList"][0]["id"])
        self.assertEqual(result.chunk_topics["topic1"]["name"], result2.chunk_topics["topic1"]["name"])
        self.assertEqual(result.status, result2.status)
        
        # Test factory methods
        success_result = L1GenerationResult.success(bio, result.clusters, result.chunk_topics)
        self.assertEqual(success_result.status, "completed")
        self.assertEqual(success_result.bio, bio)
        
        failure_result = L1GenerationResult.failure("Test error")
        self.assertEqual(failure_result.status, "failed")
        self.assertEqual(failure_result.error, "Test error")
        # Check clusters structure in failure case
        self.assertEqual(failure_result.clusters, {"clusterList": []})


if __name__ == "__main__":
    unittest.main() 