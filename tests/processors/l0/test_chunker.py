import unittest
from app.processors.l0.chunker import Chunker
from app.processors.l0.models import ChunkInfo

class TestChunker(unittest.TestCase):
    """Test cases for the Chunker class."""
    
    def setUp(self):
        self.chunker = Chunker(
            max_chunk_size=200,  # Smaller for testing
            min_chunk_size=50,
            overlap=20
        )
        self.test_document_id = "test_doc_id"
    
    def _create_chunk_info_from_dict(self, chunk_dict):
        """Helper method to convert a chunk dictionary to a ChunkInfo object for testing."""
        return ChunkInfo(
            content=chunk_dict["content"],
            chunk_id=chunk_dict["chunk_id"],
            document_id=chunk_dict["document_id"] or self.test_document_id,
            chunk_index=chunk_dict["chunk_index"],
            s3_path=f"chunks/{chunk_dict['document_id']}/chunk_{chunk_dict['chunk_index']}.txt",
            metadata=chunk_dict["metadata"]
        )
    
    def test_chunk_by_paragraph_empty_content(self):
        """Test chunking empty content."""
        chunks = self.chunker.chunk_by_paragraph("", self.test_document_id)
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_by_paragraph_single_small_content(self):
        """Test chunking content that fits in a single chunk."""
        content = "This is a small paragraph that fits in a single chunk."
        chunk_dicts = self.chunker.chunk_by_paragraph(content, self.test_document_id)
        
        # Convert dictionaries to ChunkInfo objects for testing
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, content)
        self.assertEqual(chunks[0].chunk_id, "0")
    
    def test_chunk_by_paragraph_multiple_paragraphs(self):
        """Test chunking content with multiple paragraphs."""
        content = """This is paragraph one.

This is paragraph two.

This is paragraph three.

This is paragraph four."""
        
        chunk_dicts = self.chunker.chunk_by_paragraph(content, self.test_document_id)
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        self.assertEqual(len(chunks), 1)  # Should fit in one chunk
        self.assertEqual(chunks[0].content, "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.\n\nThis is paragraph four.")
    
    def test_chunk_by_paragraph_large_paragraphs(self):
        """Test chunking content with large paragraphs."""
        # Create a large paragraph
        large_paragraph1 = "This is a very large paragraph. " * 20  # 560 chars
        large_paragraph2 = "Another large paragraph for testing. " * 20  # 680 chars
        
        content = large_paragraph1 + "\n\n" + large_paragraph2
        
        chunk_dicts = self.chunker.chunk_by_paragraph(content, self.test_document_id)
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        # Each large paragraph should be split into multiple chunks
        self.assertGreater(len(chunks), 2)
        
        # Check that no chunk exceeds max_chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), self.chunker.max_chunk_size)
    
    def test_chunk_by_fixed_size(self):
        """Test chunking content by fixed size."""
        # Create content that's larger than max_chunk_size
        content = "This is test content. " * 20  # 400 chars
        
        chunk_dicts = self.chunker.chunk_by_fixed_size(content, self.test_document_id)
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        # Should produce multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks don't exceed max_chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), self.chunker.max_chunk_size)
        
        # Check that we have overlap between chunks
        if len(chunks) > 1:
            first_chunk_end = chunks[0].content[-20:]
            second_chunk_start = chunks[1].content[:20]
            
            # At least some overlap should exist (not exact due to whitespace handling)
            self.assertTrue(
                first_chunk_end.strip() in second_chunk_start or 
                second_chunk_start.strip() in first_chunk_end
            )
    
    def test_chunk_by_paragraph_with_markdown_headers(self):
        """Test chunking content with Markdown headers."""
        content = """# Introduction
This is an introduction paragraph.

## Section 1
This is content in section 1.

## Section 2
This is content in section 2."""
        
        chunk_dicts = self.chunker.chunk_by_paragraph(content, self.test_document_id)
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        # Headers should be detected as paragraph breaks
        self.assertGreaterEqual(len(chunks), 1)
        
        # Headers should be included with their content
        self.assertTrue("# Introduction" in chunks[0].content)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        content = "This is test content."
        metadata = {"author": "Test Author"}
        document_id = "doc123"
        
        chunk_dicts = self.chunker.chunk_by_paragraph(content, document_id, metadata=metadata)
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata, metadata)
        self.assertEqual(chunks[0].document_id, document_id)
    
    def test_chunk_id_prefix(self):
        """Test that chunk_id_prefix is used correctly."""
        content = "This is test content."
        prefix = "doc_"
        
        chunk_dicts = self.chunker.chunk_by_paragraph(content, self.test_document_id, chunk_id_prefix=prefix)
        chunks = [self._create_chunk_info_from_dict(c) for c in chunk_dicts]
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, f"{prefix}0")
    
    def test_split_large_paragraph(self):
        """Test the _split_large_paragraph method."""
        large_paragraph = "This is a sentence. " * 30  # ~540 chars
        
        # Access the protected method for testing
        chunks = self.chunker._split_large_paragraph(large_paragraph)
        
        # Should split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks don't exceed max_chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.chunker.max_chunk_size)

if __name__ == '__main__':
    unittest.main() 