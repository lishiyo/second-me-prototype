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
