import logging
import os
import time
from typing import List, Optional, Dict, Any

import openai
import numpy as np

from app.processors.l0.models import ChunkInfo
from app.processors.l0.utils import retry, setup_logger

# Set up logger
logger = setup_logger(__name__)

class EmbeddingGenerator:
    """
    TODO: Replace with the new LLMService class
    Generates vector embeddings for document chunks using OpenAI's API.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-3-small",
                 batch_size: int = 10,
                 dimensions: int = 1536,
                 retry_delay: int = 2):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI embedding model to use
            batch_size: Number of texts to embed in a single API call
            dimensions: Dimensionality of the embedding vectors
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass to constructor.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.retry_delay = retry_delay
    
    @retry(max_retries=3, initial_backoff=2.0, backoff_factor=2.0, 
           error_types=(openai.RateLimitError, openai.APITimeoutError))
    def generate_embeddings(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of ChunkInfo objects to generate embeddings for
            
        Returns:
            List of ChunkInfo objects with embeddings added
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Process chunks in batches to avoid rate limits
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            
            # Extract texts from chunks, ensuring they are strings
            texts = []
            for chunk in batch:
                if not hasattr(chunk, 'content') or chunk.content is None:
                    logger.warning(f"Chunk missing content attribute or has None content")
                    texts.append("")
                elif not isinstance(chunk.content, str):
                    logger.warning(f"Non-string content found in chunk: {type(chunk.content)}")
                    texts.append(str(chunk.content))
                else:
                    texts.append(chunk.content)
            
            # Get embeddings for the batch
            response = self._get_embeddings(texts)
            
            # Update chunks with their embeddings
            for j, embedding_data in enumerate(response.data):
                if i + j < len(chunks):
                    chunks[i + j].embedding = embedding_data.embedding
            
            # Avoid rate limiting
            if i + self.batch_size < len(chunks):
                time.sleep(self.retry_delay)
        
        logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def _get_embeddings(self, texts: List[str]) -> Any:
        """
        Call OpenAI API to get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            OpenAI API response containing embeddings
        """
        try:
            return self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_for_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self._get_embeddings([text])
        return response.data[0].embedding
    
    def generate_document_embedding(self, document_content: Any) -> List[float]:
        """
        Generate a document-level embedding for the entire document.
        
        This method can accept either:
        1. A string containing the full document text, or
        2. A list of ChunkInfo objects containing chunks of the document
        
        When provided with chunks, it will create a representative embedding
        for the entire document by using a summarization approach:
        - For short documents: use the full text (truncated if needed)
        - For longer documents: use a combination of document title, first chunk, and last chunk
        
        Args:
            document_content: Either a string with full document text or a list of ChunkInfo objects
            
        Returns:
            Embedding vector as a list of floats representing the entire document
        """
        logger.info("Generating document-level embedding")
        
        # Case 1: String content
        if isinstance(document_content, str):
            # For string content, we'll truncate if it's too long
            text = document_content[:65000]  # Avoid token limits
            logger.info(f"Generating document embedding from text (length: {len(text)})")
            return self.get_embedding_for_text(text)
        
        # Case 2: List of ChunkInfo objects
        elif isinstance(document_content, list) and len(document_content) > 0 and hasattr(document_content[0], 'content'):
            chunks = document_content
            
            # For short documents with few chunks, concatenate all content
            if len(chunks) <= 3:
                combined_text = " ".join([c.content for c in chunks if c.content])
                # Truncate if too long
                combined_text = combined_text[:65000]
                logger.info(f"Generating document embedding from combined chunks (length: {len(combined_text)})")
                return self.get_embedding_for_text(combined_text)
            
            # For longer documents, use first chunk, last chunk, and any chunk with a title
            else:
                # Find a chunk that might contain the title or document metadata
                title_chunk = next((c for c in chunks if "title" in c.metadata) if chunks and hasattr(chunks[0], 'metadata') else None, None)
                
                # Combine first, title-containing (if different), and last chunk
                texts_to_combine = []
                
                # Always add first chunk
                if chunks[0].content:
                    texts_to_combine.append(chunks[0].content)
                
                # Add title chunk if it exists and isn't the first chunk
                if title_chunk and title_chunk != chunks[0] and title_chunk.content:
                    texts_to_combine.append(title_chunk.content)
                
                # Add last chunk if it's not already included
                last_chunk = chunks[-1]
                if last_chunk not in [chunks[0], title_chunk] and last_chunk.content:
                    texts_to_combine.append(last_chunk.content)
                
                # Combine texts with separation
                combined_text = " ".join(texts_to_combine)
                combined_text = combined_text[:65000]  # Truncate if too long
                
                logger.info(f"Generating document embedding from key chunks (length: {len(combined_text)})")
                return self.get_embedding_for_text(combined_text)
        
        # Handle invalid input
        else:
            error_msg = f"Invalid document_content type: {type(document_content)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
