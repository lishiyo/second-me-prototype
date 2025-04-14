import unittest
import os
import uuid
import tempfile
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional

from app.processors.l0.document_processor import DocumentProcessor
from app.processors.l0.models import (
    FileInfo, 
    ChunkInfo, 
    DocumentInsight,
    DocumentSummary,
    ProcessingStatus
)
from app.providers.blob_store import BlobStore
from app.providers.vector_db import VectorDB
from app.providers.rel_db import RelationalDB, Document, User


class TestL0Integration(unittest.TestCase):
    """Integration tests for the L0 processing pipeline."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create mock storage providers
        self.mock_blob_store = MagicMock(spec=BlobStore)
        self.mock_vector_db = MagicMock(spec=VectorDB)
        self.mock_rel_db = MagicMock(spec=RelationalDB)
        self.mock_db_session = MagicMock()
        self.mock_rel_db.get_db_session.return_value = self.mock_db_session
        
        # Create mock for Document class - this is used directly in _update_document_status_with_session
        self.document_patch = patch('app.processors.l0.document_processor.Document')
        self.mock_document_class = self.document_patch.start()
        self.mock_document = MagicMock(spec=Document)
        self.mock_document_class.return_value = self.mock_document
        
        # Create mock OpenAI API key for testing
        self.test_api_key = "test-api-key"
        
        # Setup test user
        self.test_user_id = "test-user"
        mock_user = MagicMock(spec=User)
        mock_user.id = self.test_user_id
        self.mock_rel_db.get_or_create_user.return_value = mock_user
        
        # Mock the get_document method to return None initially
        self.mock_rel_db.get_document.return_value = None
        
        # Mock the query method on db_session
        # This needs to be dynamic: return None initially, but return the mock document
        # after it's been added to the session.
        mock_query = MagicMock()
        mock_filter = MagicMock()
        
        # Store the mock document object to be returned later
        self._mock_doc_instance = None
        
        def mock_query_first(*args, **kwargs):
            # Return the mock document instance *if* it has been set (simulating it being added)
            return self._mock_doc_instance
        
        mock_filter.first.side_effect = mock_query_first
        mock_query.filter.return_value = mock_filter
        self.mock_db_session.query.return_value = mock_query
        
        # Mock the session add method to capture the added document
        def mock_session_add(obj):
            # Use the *actual* Document class for the check
            if isinstance(obj, Document):
                self._mock_doc_instance = obj # Capture the document instance when it's added
        
        self.mock_db_session.add.side_effect = mock_session_add
        
        # Create temp files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        
        # Enable API key patching (to avoid real API calls)
        self.api_key_patch = patch.dict(os.environ, {"OPENAI_API_KEY": self.test_api_key})
        self.api_key_patch.start()
        
        # Create a DocumentProcessor with our mocked dependencies
        self.processor = DocumentProcessor(
            storage_provider=self.mock_blob_store,
            vector_db_provider=self.mock_vector_db,
            rel_db_provider=self.mock_rel_db,
            openai_api_key=self.test_api_key,
            user_id=self.test_user_id
        )
        
        # Create sample documents for testing
        self.sample_text = "This is a test document for the L0 pipeline."
        self.sample_markdown = """# Test Document
        
        This is a test document with markdown formatting.
        
        ## Section 1
        
        This is the first section of the document.
        
        ## Section 2
        
        This is the second section of the document.
        """
        
        # Replace actual API calling components with mocks
        self.setup_mocks()

    def tearDown(self):
        """Clean up test environment after each test."""
        self.temp_dir.cleanup()
        self.api_key_patch.stop()
        self.document_patch.stop()

    def setup_mocks(self):
        """Configure mocks for integration testing."""
        # Mock content extraction to return test content
        self.processor.content_extractor.extract = MagicMock()
        self.processor.content_extractor.extract.side_effect = lambda file_info: file_info.content.decode('utf-8') if isinstance(file_info.content, bytes) else file_info.content
        
        # Mock document insight processor
        self.processor.document_insight_processor.process_document = MagicMock()
        self.processor.document_insight_processor.process_document.return_value = {
            "insight": DocumentInsight(
                title="Test Document",
                insight="This is a test document with multiple sections."
            ),
            "summary": DocumentSummary(
                title="Test Document",
                summary="A simple document created for testing the L0 pipeline.",
                keywords=["test", "document", "L0", "pipeline"]
            )
        }
        
        # Mock chunker to return predefined chunks
        def mock_chunking(text, document_id, **kwargs):
            # Create 2 test chunks
            return [
                {
                    "chunk_id": f"{document_id}_0",
                    "document_id": document_id,
                    "chunk_index": 0,
                    "content": "This is the first chunk.",
                    "metadata": {"document_id": document_id, "chunk_index": 0}
                },
                {
                    "chunk_id": f"{document_id}_1",
                    "document_id": document_id,
                    "chunk_index": 1,
                    "content": "This is the second chunk.",
                    "metadata": {"document_id": document_id, "chunk_index": 1}
                }
            ]
        
        self.processor.chunker.chunk_by_paragraph = MagicMock(side_effect=mock_chunking)
        self.processor.chunker.chunk_by_fixed_size = MagicMock(side_effect=mock_chunking)
        
        # Mock embedding generator
        def mock_generate_embeddings(chunks):
            # Add mock embeddings to chunks
            for chunk in chunks:
                chunk.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
            return chunks
        
        self.processor.embedding_generator.generate_embeddings = MagicMock(side_effect=mock_generate_embeddings)
        
        # Mock storage operations
        self.mock_blob_store.put_object = MagicMock(return_value=True)
        self.mock_blob_store.put_json = MagicMock(return_value=True)
        self.mock_blob_store.put_file = MagicMock(return_value=True)
        self.mock_blob_store.get_object = MagicMock(return_value=self.sample_text.encode('utf-8'))
        
        # Mock vector DB operations
        self.mock_vector_db.add_chunk = MagicMock(return_value=True)
        self.mock_vector_db.add_chunks = MagicMock(return_value=True)
        
        # Mock database operations for document creation
        mock_document = MagicMock(spec=Document)
        mock_document.id = "test-doc-id"
        self.mock_rel_db.create_document.return_value = mock_document
        self.mock_rel_db.update_document_processed.return_value = mock_document

    def create_test_file(self, content: str, filename: str = "test.txt") -> str:
        """Create a test file with the given content."""
        file_path = os.path.join(self.temp_path, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    def test_end_to_end_plain_text(self):
        """Test end-to-end processing of a plain text document."""
        # Create test file
        file_path = self.create_test_file(self.sample_text)
        
        # Create file info
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.txt",
            content_type="text/plain",
            s3_path=f"tenant/{self.test_user_id}/raw/{document_id}_test.txt",
            content=self.sample_text.encode('utf-8')
        )
        
        # Process the document
        result = self.processor.process_document(file_info)
        
        # Verify the result
        self.assertEqual(result.status, ProcessingStatus.COMPLETED)
        self.assertEqual(result.document_id, document_id)
        self.assertEqual(result.chunk_count, 2)  # From our mock chunker
        
        # Verify that all the necessary operations were performed
        # Document should be stored in blob store
        self.mock_blob_store.put_object.assert_called()
        
        # Chunks storage operation should be called - either put_json or put_object
        # The implementation might use one or the other
        self.assertTrue(
            self.mock_blob_store.put_json.called or 
            self.mock_blob_store.put_object.call_count >= 3,  # Original doc + 2 chunks
            "Neither put_json nor multiple put_object calls were made for chunks"
        )
        
        # Document should be created via the session.add method
        self.mock_db_session.add.assert_called()
        self.mock_document_class.assert_called()  # Document constructor should be called
        
        # Document should be updated to completed status
        # The implementation uses internal _update_document_status_with_session,
        # So we verify the DB session is used and the document is committed
        self.mock_rel_db.get_db_session.assert_called()
        self.mock_db_session.commit.assert_called()
        
        # Chunks should be added to vector DB
        self.assertTrue(
            self.mock_vector_db.add_chunks.called or
            self.mock_vector_db.add_chunk.call_count >= 2,
            "Neither add_chunks nor multiple add_chunk calls were made"
        )

    def test_end_to_end_markdown(self):
        """Test end-to-end processing of a markdown document."""
        # Create test file
        file_path = self.create_test_file(self.sample_markdown, "test.md")
        
        # Create file info
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.md",
            content_type="text/markdown",
            s3_path=f"tenant/{self.test_user_id}/raw/{document_id}_test.md",
            content=self.sample_markdown.encode('utf-8')
        )
        
        # Process the document
        result = self.processor.process_document(file_info)
        
        # Verify the result
        self.assertEqual(result.status, ProcessingStatus.COMPLETED)
        self.assertEqual(result.document_id, document_id)
        self.assertEqual(result.chunk_count, 2)  # From our mock chunker
        
        # Verify that insight and summary were stored
        self.assertIsNotNone(result.insight)
        self.assertEqual(result.insight.title, "Test Document")
        self.assertIsNotNone(result.summary)
        self.assertEqual(len(result.summary.keywords), 4)
        
        # Verify key operations were performed
        self.mock_blob_store.put_object.assert_called()
        self.mock_db_session.add.assert_called()  # Document is created directly
        self.mock_document_class.assert_called()  # Document constructor should be called
        self.mock_rel_db.get_db_session.assert_called()
        self.mock_db_session.commit.assert_called()

    def test_error_handling(self):
        """Test error handling during document processing."""
        # Create file info
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.txt",
            content_type="text/plain",
            s3_path=f"tenant/{self.test_user_id}/raw/{document_id}_test.txt",
            content=self.sample_text.encode('utf-8')
        )
        
        # Make content extractor raise an exception
        self.processor.content_extractor.extract.side_effect = Exception("Content extraction failed")
        
        # Process the document
        result = self.processor.process_document(file_info)
        
        # Verify the result
        self.assertEqual(result.status, ProcessingStatus.FAILED)
        self.assertEqual(result.document_id, document_id)
        self.assertIn("Content extraction failed", result.error)
        
        # Verify database operations in error case
        self.mock_rel_db.get_db_session.assert_called()
        self.mock_db_session.add.assert_called()  # Document is created directly
        self.mock_document_class.assert_called()  # Document constructor should be called
        self.mock_db_session.rollback.assert_called()     # Rollback should be called
        # Verify the provider's close_db_session is called with the mock session
        self.mock_rel_db.close_db_session.assert_called_with(self.mock_db_session)

    def test_chunking_strategies(self):
        """Test different chunking strategies."""
        # Create file info
        document_id = str(uuid.uuid4())
        file_info = FileInfo(
            document_id=document_id,
            filename="test.txt",
            content_type="text/plain",
            s3_path=f"tenant/{self.test_user_id}/raw/{document_id}_test.txt",
            content=self.sample_text.encode('utf-8')
        )
        
        # Test paragraph chunking
        self.processor.chunking_strategy = "paragraph"
        result1 = self.processor.process_document(file_info)
        self.assertEqual(result1.status, ProcessingStatus.COMPLETED)
        self.processor.chunker.chunk_by_paragraph.assert_called()
        self.processor.chunker.chunk_by_fixed_size.assert_not_called()
        
        # Reset call counts
        self.processor.chunker.chunk_by_paragraph.reset_mock()
        self.processor.chunker.chunk_by_fixed_size.reset_mock()
        
        # Test fixed-size chunking
        self.processor.chunking_strategy = "fixed"
        result2 = self.processor.process_document(file_info)
        self.assertEqual(result2.status, ProcessingStatus.COMPLETED)
        self.processor.chunker.chunk_by_fixed_size.assert_called()
        self.processor.chunker.chunk_by_paragraph.assert_not_called()


if __name__ == "__main__":
    unittest.main() 