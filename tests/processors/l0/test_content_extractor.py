import unittest
from unittest.mock import MagicMock, patch
import io
import os

from app.processors.l0.content_extractor import ContentExtractor
from app.processors.l0.models import FileInfo

class TestContentExtractor(unittest.TestCase):
    """Test cases for the ContentExtractor class."""
    
    def setUp(self):
        self.extractor = ContentExtractor()
    
    def test_text_extraction(self):
        """Test extracting content from a text file."""
        content = b"This is a sample text document."
        file_info = FileInfo(
            document_id="doc123",
            filename="sample.txt",
            content_type="text/plain",
            s3_path="tenant/1/raw/doc123.txt",
            content=content
        )
        
        extracted_text = self.extractor.extract(file_info)
        self.assertEqual(extracted_text, "This is a sample text document.")
    
    def test_none_content(self):
        """Test handling of None content."""
        file_info = FileInfo(
            document_id="doc123",
            filename="sample.txt",
            content_type="text/plain",
            s3_path="tenant/1/raw/doc123.txt",
            content=None
        )
        
        with self.assertRaises(ValueError):
            self.extractor.extract(file_info)
    
    @patch('docx.Document')
    def test_docx_extraction(self, mock_docx):
        """Test extracting content from a DOCX file."""
        # Mock the Document class
        mock_doc = MagicMock()
        mock_paragraphs = [
            MagicMock(text="Paragraph 1"),
            MagicMock(text="Paragraph 2"),
            MagicMock(text=""),  # Empty paragraph
            MagicMock(text="Paragraph 3")
        ]
        mock_doc.paragraphs = mock_paragraphs
        mock_docx.return_value = mock_doc
        
        content = b"docx file binary content"
        file_info = FileInfo(
            document_id="doc123",
            filename="sample.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            s3_path="tenant/1/raw/doc123.docx",
            content=content
        )
        
        extracted_text = self.extractor.extract(file_info)
        self.assertEqual(extracted_text, "Paragraph 1\nParagraph 2\nParagraph 3")
        
        # Verify Document was called with a BytesIO object
        mock_docx.assert_called_once()
        call_args = mock_docx.call_args[0][0]
        self.assertIsInstance(call_args, io.BytesIO)
    
    @patch('PyPDF2.PdfReader')
    def test_pdf_extraction(self, mock_pdf_reader):
        """Test extracting content from a PDF file."""
        # Mock the PdfReader class
        mock_reader = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader
        
        content = b"pdf file binary content"
        file_info = FileInfo(
            document_id="doc123",
            filename="sample.pdf",
            content_type="application/pdf",
            s3_path="tenant/1/raw/doc123.pdf",
            content=content
        )
        
        extracted_text = self.extractor.extract(file_info)
        self.assertEqual(extracted_text, "Page 1 content\n\nPage 2 content")
        
        # Verify PdfReader was called with a BytesIO object
        mock_pdf_reader.assert_called_once()
        call_args = mock_pdf_reader.call_args[0][0]
        self.assertIsInstance(call_args, io.BytesIO)
    
    def test_unsupported_content_type(self):
        """Test handling of unsupported content types."""
        content = b"Binary content"
        file_info = FileInfo(
            document_id="doc123",
            filename="sample.bin",
            content_type="application/octet-stream",
            s3_path="tenant/1/raw/doc123.bin",
            content=content
        )
        
        # Should fall back to text extractor with a warning
        extracted_text = self.extractor.extract(file_info)
        # It will try to decode binary as UTF-8, which might result in garbage or decode errors
        self.assertIsInstance(extracted_text, str)
    
    def test_unicode_decode_error(self):
        """Test handling of unicode decode errors."""
        # Create content that will cause a decode error
        content = b"\x80\x81\x82\x83"
        file_info = FileInfo(
            document_id="doc123",
            filename="sample.txt",
            content_type="text/plain",
            s3_path="tenant/1/raw/doc123.txt",
            content=content
        )
        
        # Should handle the error and return a string with replacement characters
        extracted_text = self.extractor.extract(file_info)
        self.assertIsInstance(extracted_text, str)
        self.assertNotEqual(extracted_text, "")

if __name__ == '__main__':
    unittest.main() 