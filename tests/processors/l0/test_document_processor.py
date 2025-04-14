import unittest
from unittest.mock import MagicMock, patch
import uuid
from typing import Dict, Any, List, Optional

from app.processors.l0.document_processor import DocumentProcessor
from app.processors.l0.models import (
    FileInfo, 
    ChunkInfo, 
    DocumentInsight,
    DocumentSummary,
    ProcessingStatus
)

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the DocumentProcessor class."""
    
    def setUp(self):
        # Create mock storage and vector DB providers
        self.mock_storage = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_rel_db = MagicMock()
        self.mock_db_session = MagicMock()
        self.mock_rel_db.get_db_session.return_value = self.mock_db_session
        
        # Create mock OpenAI API key for testing
        self.mock_api_key = "test-api-key"
        
        # Create the document processor with mocked dependencies
        self.processor = DocumentProcessor(
            storage_provider=self.mock_storage,
            vector_db_provider=self.mock_vector_db,
            rel_db_provider=self.mock_rel_db,
            openai_api_key=self.mock_api_key
        )
        
        # Mock all the internal components
        self.processor.content_extractor = MagicMock()
        self.processor.chunker = MagicMock()
        self.processor.document_insight_processor = MagicMock()
        self.processor.embedding_generator = MagicMock()
    
    def test_process_document_success(self):
        """Test successful document processing flow."""
        # Create test data
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.pdf",
            content_type="application/pdf",
            s3_path=f"tenant/1/raw/{document_id}.pdf",
            content=b"test content"
        )
        
        # Configure mocks
        self.mock_storage.store_document.return_value = file_info.s3_path
        self.mock_storage.put_object.return_value = file_info.s3_path
        self.mock_storage.put_file.return_value = file_info.s3_path
        
        # Mock content extraction
        extracted_content = "Extracted text content"
        self.processor.content_extractor.extract.return_value = extracted_content
        
        # Mock document analysis
        insight = DocumentInsight(
            title="Test Document",
            insight="This is a test document insight."
        )
        
        summary = DocumentSummary(
            title="Test Document",
            summary="This is a test document summary.",
            keywords=["test", "document", "processing"]
        )
        
        analysis_results = {
            "insight": insight,
            "summary": summary
        }
        self.processor.document_insight_processor.process_document.return_value = analysis_results
        
        # Mock chunking to return dictionaries, not ChunkInfo objects
        chunk_dicts = [
            {
                "chunk_id": f"{document_id}_0",
                "document_id": document_id,
                "chunk_index": 0,
                "content": "Chunk 1 content",
                "metadata": {"document_id": document_id}
            },
            {
                "chunk_id": f"{document_id}_1",
                "document_id": document_id,
                "chunk_index": 1,
                "content": "Chunk 2 content",
                "metadata": {"document_id": document_id}
            }
        ]
        self.processor.chunker.chunk_by_paragraph.return_value = chunk_dicts
        
        # Create corresponding ChunkInfo objects for the embedding mock
        chunks = [
            ChunkInfo(
                chunk_id=chunk_dict["chunk_id"],
                document_id=chunk_dict["document_id"],
                chunk_index=chunk_dict["chunk_index"],
                content=chunk_dict["content"],
                s3_path="",
                metadata=chunk_dict["metadata"]
            )
            for chunk_dict in chunk_dicts
        ]
        
        # Mock embedding generation
        chunks_with_embeddings = chunks.copy()
        for chunk in chunks_with_embeddings:
            chunk.embedding = [0.1, 0.2, 0.3]  # Mock embedding
        self.processor.embedding_generator.generate_embeddings.return_value = chunks_with_embeddings
        
        # Process the document
        result = self.processor.process_document(file_info)
        
        # Verify the result
        self.assertEqual(result.status, ProcessingStatus.COMPLETED)
        self.assertEqual(result.document_id, document_id)
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.insight, insight)
        self.assertEqual(result.summary, summary)
        
        # Verify all components were called correctly
        self.processor.content_extractor.extract.assert_called_once_with(file_info)
        self.processor.document_insight_processor.process_document.assert_called_once()
        self.processor.embedding_generator.generate_embeddings.assert_called()
        self.mock_vector_db.add_chunk.assert_called()
    
    def test_process_document_failure(self):
        """Test document processing with failure."""
        # Create test data
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.pdf",
            content_type="application/pdf",
            s3_path=f"tenant/1/raw/{document_id}.pdf",
            content=b"test content"
        )
        
        # Make content extraction raise an exception
        error_message = "Failed to extract content"
        self.processor.content_extractor.extract.side_effect = Exception(error_message)
        
        # Process the document
        result = self.processor.process_document(file_info)
        
        # Verify the result
        self.assertEqual(result.status, ProcessingStatus.FAILED)
        self.assertEqual(result.document_id, document_id)
        self.assertEqual(result.error, error_message)
        
        # Verify only the storage and extraction were called
        self.processor.content_extractor.extract.assert_called_once_with(file_info)
        
        # Verify other components were not called
        self.processor.document_insight_processor.process_document.assert_not_called()
        self.processor.chunker.chunk_by_paragraph.assert_not_called()
        self.processor.embedding_generator.generate_embeddings.assert_not_called()
        self.mock_vector_db.add_chunk.assert_not_called()
    
    def test_chunking_strategy(self):
        """Test different chunking strategies."""
        # Create test data
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.txt",
            content_type="text/plain",
            s3_path=f"tenant/1/raw/{document_id}.txt",
            content=b"test content"
        )
        
        # Configure processor to use fixed-size chunking
        self.processor.chunking_strategy = "fixed"
        
        # Set up minimal successful mocks
        self.processor.content_extractor.extract.return_value = "Extracted text"
        
        insight = DocumentInsight(title="Test", insight="Test insight")
        summary = DocumentSummary(title="Test", summary="Test summary", keywords=["test"])
        self.processor.document_insight_processor.process_document.return_value = {
            "insight": insight,
            "summary": summary
        }
        
        # Mock chunking to return dictionary, not a ChunkInfo object
        chunk_dict = {
            "chunk_id": "1",
            "document_id": document_id,
            "chunk_index": 0,
            "content": "Chunk",
            "metadata": {"document_id": document_id}
        }
        self.processor.chunker.chunk_by_fixed_size.return_value = [chunk_dict]
        
        # Create the corresponding ChunkInfo for the embedding mock
        chunk = ChunkInfo(
            chunk_id=chunk_dict["chunk_id"],
            document_id=chunk_dict["document_id"],
            chunk_index=chunk_dict["chunk_index"],
            content=chunk_dict["content"],
            s3_path="",
            metadata=chunk_dict["metadata"]
        )
        
        # Add embedding
        chunk.embedding = [0.1, 0.2, 0.3]
        self.processor.embedding_generator.generate_embeddings.return_value = [chunk]
        
        # Process the document
        self.processor.process_document(file_info)
        
        # Verify fixed-size chunking was called instead of paragraph chunking
        self.processor.chunker.chunk_by_fixed_size.assert_called_once()
        self.processor.chunker.chunk_by_paragraph.assert_not_called()

if __name__ == '__main__':
    unittest.main() 