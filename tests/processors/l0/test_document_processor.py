import unittest
from unittest.mock import MagicMock, patch
import uuid

from app.processors.l0.document_processor import DocumentProcessor
from app.processors.l0.models import (
    FileInfo, 
    ChunkInfo, 
    DocumentInsights,
    ProcessingResult,
    ProcessingStatus
)

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the DocumentProcessor class."""
    
    def setUp(self):
        # Create mock storage and vector DB providers
        self.mock_storage = MagicMock()
        self.mock_vector_db = MagicMock()
        
        # Create mock OpenAI API key for testing
        self.mock_api_key = "test-api-key"
        
        # Create the document processor with mocked dependencies
        self.processor = DocumentProcessor(
            storage_provider=self.mock_storage,
            vector_db_provider=self.mock_vector_db,
            openai_api_key=self.mock_api_key
        )
        
        # Mock all the internal components
        self.processor.content_extractor = MagicMock()
        self.processor.chunker = MagicMock()
        self.processor.document_analyzer = MagicMock()
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
        
        # Mock content extraction
        extracted_content = "Extracted text content"
        self.processor.content_extractor.extract.return_value = extracted_content
        
        # Mock document analysis
        insights = DocumentInsights(
            title="Test Document",
            summary="This is a test document summary.",
            keywords=["test", "document", "processing"]
        )
        self.processor.document_analyzer.analyze.return_value = insights
        
        # Mock chunking
        chunks = [
            ChunkInfo(
                chunk_id=f"{document_id}_0",
                document_id=document_id,
                chunk_index=0,
                content="Chunk 1 content",
                s3_path=f"chunks/{document_id}/chunk_0.txt",
                metadata={"document_id": document_id}
            ),
            ChunkInfo(
                chunk_id=f"{document_id}_1",
                document_id=document_id,
                chunk_index=1,
                content="Chunk 2 content",
                s3_path=f"chunks/{document_id}/chunk_1.txt",
                metadata={"document_id": document_id}
            )
        ]
        self.processor.chunker.chunk_by_paragraph.return_value = chunks
        
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
        self.assertEqual(result.insights, insights)
        
        # Verify all components were called correctly
        self.mock_storage.store_document.assert_called_once_with(file_info)
        self.processor.content_extractor.extract.assert_called_once_with(file_info)
        self.processor.document_analyzer.analyze.assert_called_once_with(extracted_content, file_info.filename)
        self.processor.chunker.chunk_by_paragraph.assert_called_once()
        self.processor.embedding_generator.generate_embeddings.assert_called_once_with(chunks)
        self.mock_vector_db.store_embeddings.assert_called_once_with(chunks_with_embeddings)
    
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
        self.mock_storage.store_document.assert_called_once_with(file_info)
        self.processor.content_extractor.extract.assert_called_once_with(file_info)
        
        # Verify other components were not called
        self.processor.document_analyzer.analyze.assert_not_called()
        self.processor.chunker.chunk_by_paragraph.assert_not_called()
        self.processor.embedding_generator.generate_embeddings.assert_not_called()
        self.mock_vector_db.store_embeddings.assert_not_called()
    
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
        self.processor.document_analyzer.analyze.return_value = DocumentInsights(
            title="Test", summary="Test", keywords=["test"]
        )
        chunks = [ChunkInfo(
            chunk_id="1", document_id=document_id, chunk_index=0,
            content="Chunk", s3_path="path"
        )]
        self.processor.chunker.chunk_by_fixed_size.return_value = chunks
        self.processor.embedding_generator.generate_embeddings.return_value = chunks
        
        # Process the document
        self.processor.process_document(file_info)
        
        # Verify fixed-size chunking was called instead of paragraph chunking
        self.processor.chunker.chunk_by_fixed_size.assert_called_once()
        self.processor.chunker.chunk_by_paragraph.assert_not_called()

if __name__ == '__main__':
    unittest.main() 