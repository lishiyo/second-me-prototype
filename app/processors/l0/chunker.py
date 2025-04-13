import re
from typing import List, Optional, Dict, Any

from app.processors.l0.models import ChunkInfo
from app.processors.l0.utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

class Chunker:
    """
    Divides document content into smaller, manageable chunks
    based on different strategies (paragraph, fixed-size, semantic).
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000, 
                 min_chunk_size: int = 100,
                 overlap: int = 50):
        """
        Initialize the chunker with configurable parameters.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            min_chunk_size: Minimum size of a chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        # Patterns for detecting paragraph and section breaks
        self.paragraph_delimiters = [
            r'\n\s*\n',      # Double newlines
            r'\n#{1,6}\s',   # Markdown headers
            r'\n\*\*\*+\s*', # Markdown horizontal rules
        ]
    
    def chunk_by_paragraph(self, content: str, document_id: str = "", 
                         chunk_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split content by paragraphs and natural breaks.
        
        Args:
            content: Text content to be chunked
            document_id: ID of the document being chunked
            chunk_id_prefix: Optional prefix for chunk IDs
            metadata: Optional metadata about the document
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not content:
            logger.warning("Cannot chunk empty content")
            return []
        
        # Create a combined pattern for paragraph detection
        pattern = '|'.join(self.paragraph_delimiters)
        
        # Split the content by the combined pattern
        paragraphs = re.split(pattern, content)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            logger.warning("No paragraphs found in content after splitting")
            # Create a single chunk if no paragraphs are detected
            return [self._create_chunk_dict(content.strip(), 0, document_id, chunk_id_prefix, metadata)]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # If paragraph exceeds maximum size, split it further
            if para_size > self.max_chunk_size:
                # If there's content in current_chunk, add it as a chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk_dict(
                        chunk_text, chunk_idx, document_id, chunk_id_prefix, metadata
                    ))
                    chunk_idx += 1
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph into smaller chunks
                sub_chunks = self._split_large_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk_dict(
                        sub_chunk, chunk_idx, document_id, chunk_id_prefix, metadata
                    ))
                    chunk_idx += 1
            
            # If adding paragraph exceeds max size, create a new chunk
            elif current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk_dict(
                    chunk_text, chunk_idx, document_id, chunk_id_prefix, metadata
                ))
                chunk_idx += 1
                current_chunk = [paragraph]
                current_size = para_size
            
            # Otherwise add paragraph to current chunk
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        # Add any remaining content as the final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk_dict(
                chunk_text, chunk_idx, document_id, chunk_id_prefix, metadata
            ))
        
        logger.info(f"Created {len(chunks)} chunks from content")
        return chunks
    
    def chunk_by_fixed_size(self, content: str, document_id: str = "",
                          chunk_id_prefix: str = "", metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split content into fixed-size chunks with optional overlap.
        
        Args:
            content: Text content to be chunked
            document_id: ID of the document being chunked
            chunk_id_prefix: Optional prefix for chunk IDs
            metadata: Optional metadata about the document
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not content:
            logger.warning("Cannot chunk empty content")
            return []
        
        content = content.strip()
        content_length = len(content)
        
        # If content is smaller than min chunk size, return as a single chunk
        if content_length <= self.min_chunk_size:
            return [self._create_chunk_dict(content, 0, document_id, chunk_id_prefix, metadata)]
        
        chunks = []
        chunk_idx = 0
        start_pos = 0
        
        while start_pos < content_length:
            end_pos = min(start_pos + self.max_chunk_size, content_length)
            
            # Try to find a good break point (whitespace) near the end position
            if end_pos < content_length:
                # Look for the last whitespace within the last 20% of the chunk
                search_start = max(start_pos, end_pos - int(self.max_chunk_size * 0.2))
                nearest_break = content.rfind(' ', search_start, end_pos)
                
                if nearest_break != -1:
                    end_pos = nearest_break
            
            # Extract the chunk
            chunk_text = content[start_pos:end_pos].strip()
            
            if chunk_text:
                chunks.append(self._create_chunk_dict(
                    chunk_text, chunk_idx, document_id, chunk_id_prefix, metadata
                ))
                chunk_idx += 1
            
            # Move start position for next chunk, considering overlap
            start_pos = end_pos - self.overlap if self.overlap < end_pos - start_pos else end_pos
        
        logger.info(f"Created {len(chunks)} fixed-size chunks from content")
        return chunks
    
    def _create_chunk_dict(self, content: str, chunk_idx: int, document_id: str,
                         chunk_id_prefix: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Helper method to create a chunk dictionary with all necessary information.
        
        Args:
            content: Chunk content
            chunk_idx: Index of the chunk
            document_id: ID of the document
            chunk_id_prefix: Optional prefix for chunk ID
            metadata: Optional metadata
            
        Returns:
            Dictionary with chunk information
        """
        # Generate chunk_id using prefix and index
        chunk_id = f"{chunk_id_prefix}{chunk_idx}" if chunk_id_prefix else str(chunk_idx)
        
        # Ensure content is a string
        if not isinstance(content, str):
            logger.warning(f"Non-string content detected in chunk {chunk_idx}: {type(content)}")
            content = str(content) if content is not None else ""
        
        # Create a dictionary with all chunk information
        chunk_dict = {
            "content": content,
            "chunk_id": chunk_id,
            "chunk_index": chunk_idx,
            "document_id": document_id,
            "metadata": metadata or {}
        }
        
        return chunk_dict
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """
        Split a large paragraph into smaller chunks by sentences.
        
        Args:
            paragraph: Large paragraph to split
            
        Returns:
            List of smaller chunks
        """
        # Simple sentence splitting pattern
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If single sentence exceeds max size, split by fixed size
            if sentence_size > self.max_chunk_size:
                # Add any content in current_chunk first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split the large sentence by fixed size
                for i in range(0, sentence_size, self.max_chunk_size - self.overlap):
                    end_pos = min(i + self.max_chunk_size, sentence_size)
                    chunks.append(sentence[i:end_pos])
            
            # If adding sentence exceeds max size, create a new chunk
            elif current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            
            # Otherwise add sentence to current chunk
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add any remaining content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
