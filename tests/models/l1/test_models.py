import unittest
import numpy as np
from datetime import datetime
from app.models.l1 import (
    Note, Chunk, Topic, Cluster, Memory, 
    L1Shade, ShadeInfo, ShadeMergeInfo, MergedShadeResult,
    Bio, L1GenerationResult, ShadeTimeline
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
            descThirdView="Description in third view",
            contentThirdView="Content in third view",
            descSecondView="Description in second view",
            contentSecondView="Content in second view"
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
        self.assertEqual(shade_info.descThirdView, shade_info2.descThirdView)
        self.assertEqual(shade_info.contentThirdView, shade_info2.contentThirdView)
        self.assertEqual(shade_info.descSecondView, shade_info2.descSecondView)
        self.assertEqual(shade_info.contentSecondView, shade_info2.contentSecondView)
    
    def test_shade(self):
        """Test L1Shade model."""
        # Create a shade
        shade = L1Shade(
            id="shade1",
            name="Test Shade",
            summary="This is a test shade",
            aspect="Personality",
            icon="🧠",
            descThirdView="Description in third view",
            contentThirdView="Content in third view",
            descSecondView="Description in second view",
            contentSecondView="Content in second view",
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
        self.assertEqual(shade.descThirdView, shade2.descThirdView)
        self.assertEqual(shade.contentThirdView, shade2.contentThirdView)
        self.assertEqual(shade.descSecondView, shade2.descSecondView)
        self.assertEqual(shade.contentSecondView, shade2.contentSecondView)
        self.assertEqual(shade.confidence, shade2.confidence)
        self.assertEqual(shade.metadata, shade2.metadata)
    
    def test_shade_merge_info(self):
        """Test ShadeMergeInfo model."""
        # Create an instance of ShadeMergeInfo
        shade_merge_info = ShadeMergeInfo(
            shade_id="shade_id_1",
            name="Shade Name",
            aspect="Personal Growth",
            icon="🌱",
            descThirdView="This person is characterized by...",
            contentThirdView="Detailed information about the shade...",
            descSecondView="You are characterized by...",
            contentSecondView="Detailed information about you...",
            cluster_info={"cluster_id": "cluster123", "notes": ["note1", "note2"]},
            metadata={"source": "test", "confidence": 0.9},
            timelines=[
                ShadeTimeline(
                    refMemoryId="memory1",
                    createTime="1234567890",
                    descThirdView="In 2021, they started...",
                    descSecondView="In 2021, you started..."
                )
            ]
        )

        # Convert to dict
        shade_merge_info_dict = shade_merge_info.to_dict()

        # Verify that the dict contains the correct values
        assert shade_merge_info_dict["shade_id"] == "shade_id_1"
        assert shade_merge_info_dict["name"] == "Shade Name"
        assert shade_merge_info_dict["aspect"] == "Personal Growth"
        assert shade_merge_info_dict["icon"] == "🌱"
        assert shade_merge_info_dict["descThirdView"] == "This person is characterized by..."
        assert shade_merge_info_dict["contentThirdView"] == "Detailed information about the shade..."
        assert shade_merge_info_dict["descSecondView"] == "You are characterized by..."
        assert shade_merge_info_dict["contentSecondView"] == "Detailed information about you..."
        assert shade_merge_info_dict["cluster_info"]["cluster_id"] == "cluster123"
        assert shade_merge_info_dict["metadata"]["source"] == "test"
        assert len(shade_merge_info_dict["timelines"]) == 1
        assert shade_merge_info_dict["timelines"][0]["refMemoryId"] == "memory1"

        # Create from dict
        shade_merge_info_2 = ShadeMergeInfo.from_dict(shade_merge_info_dict)

        # Verify that the new instance has the correct values
        assert shade_merge_info_2.shade_id == "shade_id_1"
        assert shade_merge_info_2.name == "Shade Name"
        assert shade_merge_info_2.aspect == "Personal Growth"
        assert shade_merge_info_2.icon == "🌱"
        assert shade_merge_info_2.descThirdView == "This person is characterized by..."
        assert shade_merge_info_2.contentThirdView == "Detailed information about the shade..."
        assert shade_merge_info_2.descSecondView == "You are characterized by..."
        assert shade_merge_info_2.contentSecondView == "Detailed information about you..."
        assert shade_merge_info_2.cluster_info["cluster_id"] == "cluster123"
        assert shade_merge_info_2.metadata["source"] == "test"
        assert len(shade_merge_info_2.timelines) == 1
        assert shade_merge_info_2.timelines[0].refMemoryId == "memory1"
        
        # Test to_json method
        json_dict = shade_merge_info.to_json()
        assert json_dict["id"] == "shade_id_1"
        assert json_dict["name"] == "Shade Name"
        assert json_dict["aspect"] == "Personal Growth"
        assert json_dict["icon"] == "🌱"
        assert json_dict["descThirdView"] == "This person is characterized by..."
        assert json_dict["contentThirdView"] == "Detailed information about the shade..."
        assert json_dict["descSecondView"] == "You are characterized by..."
        assert json_dict["contentSecondView"] == "Detailed information about you..."
        assert "timelines" in json_dict
        assert len(json_dict["timelines"]) == 1
        
        # Test helper methods
        shade_merge_info.improve_shade_info("Improved description", "Improved content")
        assert shade_merge_info.descThirdView == "Improved description"
        assert shade_merge_info.contentThirdView == "Improved content"
        
        shade_merge_info.add_second_view("New second view desc", "New second view content")
        assert shade_merge_info.descSecondView == "New second view desc"
        assert shade_merge_info.contentSecondView == "New second view content"
        
        # Test to_str method
        str_output = shade_merge_info.to_str()
        assert "**[Name]**: Shade Name" in str_output
        assert "**[Aspect]**: Personal Growth" in str_output
        assert "**[Icon]**: 🌱" in str_output
        assert "**[Description]**: \nImproved description" in str_output
        assert "**[Content]**: \nImproved content" in str_output
        assert "**[Cluster Info]**:" in str_output
        assert "memory1" in str_output
    
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
        
        # Test to_dict method
        merged_shade_result_dict = merged_shade_result.to_dict()
        assert merged_shade_result_dict["success"] == True
        assert len(merged_shade_result_dict["merge_shade_list"]) == 2
        assert merged_shade_result_dict["merge_shade_list"][0]["id"] == "merged1"
        assert merged_shade_result_dict["merge_shade_list"][0]["name"] == "Merged Shade 1"
        assert merged_shade_result_dict["merge_shade_list"][1]["id"] == "merged2"
        assert merged_shade_result_dict["error"] is None
        
        # Test from_dict method
        merged_shade_result2 = MergedShadeResult.from_dict(merged_shade_result_dict)
        assert merged_shade_result2.success == True
        assert len(merged_shade_result2.merge_shade_list) == 2
        assert merged_shade_result2.merge_shade_list[0]["id"] == "merged1"
        assert merged_shade_result2.merge_shade_list[0]["name"] == "Merged Shade 1"
        assert merged_shade_result2.merge_shade_list[1]["id"] == "merged2"
        assert merged_shade_result2.error is None
        
        # Test with error
        error_result = MergedShadeResult(
            success=False,
            merge_shade_list=[],
            error="Failed to merge shades"
        )
        
        error_dict = error_result.to_dict()
        assert error_dict["success"] == False
        assert len(error_dict["merge_shade_list"]) == 0
        assert error_dict["error"] == "Failed to merge shades"
        
        # Test from dict with error
        error_result2 = MergedShadeResult.from_dict(error_dict)
        assert error_result2.success == False
        assert len(error_result2.merge_shade_list) == 0
        assert error_result2.error == "Failed to merge shades"
    
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

    def test_shade_timeline(self):
        """Test the ShadeTimeline class."""
        # Create a timeline entry
        timeline = ShadeTimeline(
            refMemoryId="mem_123",
            createTime="2023-01-01T12:00:00",
            descSecondView="You went to the park yesterday.",
            descThirdView="User went to the park yesterday.",
            isNew=True
        )
        
        # Test property getters
        self.assertEqual(timeline.refMemoryId, "mem_123")
        self.assertEqual(timeline.createTime, "2023-01-01T12:00:00")
        self.assertEqual(timeline.descSecondView, "You went to the park yesterday.")
        self.assertEqual(timeline.descThirdView, "User went to the park yesterday.")
        self.assertEqual(timeline.isNew, True)
        
        # Test conversion to dictionary
        timeline_dict = timeline.to_dict()
        self.assertEqual(timeline_dict["refMemoryId"], "mem_123")
        self.assertEqual(timeline_dict["createTime"], "2023-01-01T12:00:00")
        self.assertEqual(timeline_dict["description"], "User went to the park yesterday.")
        self.assertEqual(timeline_dict["description_second_view"], "You went to the park yesterday.")
        self.assertTrue(timeline_dict["isNew"])
        self.assertEqual(timeline_dict["refId"], "mem_123")  # Should include both for compatibility
        
        # Test conversion to JSON (lpm_kernel format)
        timeline_json = timeline.to_json()
        self.assertEqual(timeline_json["refMemoryId"], "mem_123")
        self.assertEqual(timeline_json["createTime"], "2023-01-01T12:00:00")
        self.assertEqual(timeline_json["descThirdView"], "User went to the park yesterday.")
        self.assertEqual(timeline_json["descSecondView"], "You went to the park yesterday.")
        
        # Test creation from raw format dictionary with our field naming
        raw_data = {
            "refMemoryId": "mem_456",
            "createTime": "2023-02-01T12:00:00",
            "description": "User called a friend.",
            "description_second_view": "You called a friend.",
            "isNew": False
        }
        timeline2 = ShadeTimeline.from_raw_format(raw_data)
        self.assertEqual(timeline2.refMemoryId, "mem_456")
        self.assertEqual(timeline2.descThirdView, "User called a friend.")
        self.assertEqual(timeline2.descSecondView, "You called a friend.")
        self.assertFalse(timeline2.isNew)
        
        # Test creation from raw format dictionary with lpm_kernel's field naming
        raw_data2 = {
            "refId": "mem_789",
            "createTime": "2023-03-01T12:00:00",
            "descThirdView": "User wrote an email.",
            "descSecondView": "You wrote an email.",
            "isNew": True
        }
        timeline3 = ShadeTimeline.from_raw_format(raw_data2)
        self.assertEqual(timeline3.refMemoryId, "mem_789")
        self.assertEqual(timeline3.descThirdView, "User wrote an email.")
        self.assertEqual(timeline3.descSecondView, "You wrote an email.")
        
        # Test add_second_view method
        timeline4 = ShadeTimeline(
            refMemoryId="mem_999",
            createTime="2023-04-01T12:00:00",
            descThirdView="User went shopping."
        )
        timeline4.add_second_view("You went shopping.")
        self.assertEqual(timeline4.descSecondView, "You went shopping.")


if __name__ == "__main__":
    unittest.main() 