import io
from typing import Optional, Dict, Any

import docx
import PyPDF2
from pptx import Presentation

from app.processors.l0.models import FileInfo
from app.processors.l0.utils import retry, safe_execute, setup_logger

# Set up logger
logger = setup_logger(__name__)

class ContentExtractor:
    """
    Extracts text content from various file types.
    Supports: PDF, DOCX, PPTX, TXT and other text-based formats.
    """
    
    def __init__(self):
        self.extractors = {
            'application/pdf': self._extract_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': self._extract_pptx,
            'text/plain': self._extract_text,
            'text/markdown': self._extract_text,
            'text/html': self._extract_text,
            'application/json': self._extract_text,
        }
    
    def extract(self, file_info: FileInfo) -> str:
        """
        Extract content from a file based on its content type.
        
        Args:
            file_info: FileInfo object containing file metadata and content
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file content is None or content type is unsupported
        """
        if file_info.content is None:
            raise ValueError("File content is None")
        
        extractor = self.extractors.get(file_info.content_type)
        
        if extractor is None:
            logger.warning(f"No specific extractor for content type: {file_info.content_type}. Trying text extractor.")
            extractor = self._extract_text
        
        logger.info(f"Extracting content from {file_info.filename} ({file_info.content_type})")
        return extractor(file_info)
    
    @retry(max_retries=2)
    def _extract_pdf(self, file_info: FileInfo) -> str:
        """Extract text from PDF files."""
        pdf_content = []
        
        with io.BytesIO(file_info.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = safe_execute(page.extract_text, default_value="")
                
                if text:
                    pdf_content.append(text)
                else:
                    logger.warning(f"Could not extract text from page {page_num} in {file_info.filename}")
        
        return "\n\n".join(pdf_content)
    
    def _extract_docx(self, file_info: FileInfo) -> str:
        """Extract text from DOCX files."""
        with io.BytesIO(file_info.content) as docx_file:
            doc = docx.Document(docx_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
    
    def _extract_pptx(self, file_info: FileInfo) -> str:
        """Extract text from PPTX files."""
        content = []
        
        with io.BytesIO(file_info.content) as pptx_file:
            presentation = Presentation(pptx_file)
            
            for i, slide in enumerate(presentation.slides, 1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_text.append(shape.text)
                
                if slide_text:
                    content.append(f"Slide {i}:\n" + "\n".join(slide_text))
        
        return "\n\n".join(content)
    
    def _extract_text(self, file_info: FileInfo) -> str:
        """Extract text from plain text files."""
        try:
            return file_info.content.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {file_info.filename}, trying with errors='replace'")
            return file_info.content.decode('utf-8', errors='replace')
