"""
LLMService for handling interactions with LLM models.

This module provides a service for interacting with LLM models,
handling retries, and providing utility functions for LLM-based generation.
"""
import logging
import os
import time
from typing import List, Dict, Any, Optional, Union

import openai

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for LLM interactions.
    
    This class provides methods for making LLM API calls,
    handling retries, and providing utility functions for
    LLM-based generation.
    
    Attributes:
        api_key: OpenAI API key
        default_model: Default model to use for completions
        max_retries: Maximum number of retries for failed API calls
        retry_delay: Delay between retries in seconds
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the LLMService.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            default_model: Default model to use for completions
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not provided, API calls will fail")
            
        openai.api_key = self.api_key
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Call the OpenAI Chat Completion API with retries.
        
        Args:
            messages: List of message dictionaries (role, content)
            model: OpenAI model to use (defaults to self.default_model)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Stop sequences
            
        Returns:
            API response dictionary
        """
        model = model or self.default_model
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making OpenAI API call (attempt {attempt + 1}/{self.max_retries})")
                
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop
                )
                
                return response
                
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Generate embeddings for text using OpenAI's embedding API.
        
        Args:
            texts: Text or list of texts to embed
            model: OpenAI embedding model to use
            
        Returns:
            List of embeddings (float vectors)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Generating embeddings for {len(texts)} texts (attempt {attempt + 1})")
                
                response = openai.Embedding.create(
                    model=model,
                    input=texts
                )
                
                # Extract embeddings from response
                embeddings = [data["embedding"] for data in response["data"]]
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Embedding generation failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise
    
    def summarize_text(
        self,
        text: str,
        max_length: int = 150,
        temperature: float = 0.5
    ) -> str:
        """
        Summarize text to a specified maximum length.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            temperature: Sampling temperature
            
        Returns:
            Summarized text
        """
        messages = [
            {"role": "system", "content": f"Summarize the following text in no more than {max_length} words."},
            {"role": "user", "content": text}
        ]
        
        response = self.chat_completion(messages, temperature=temperature)
        
        return response["choices"][0]["message"]["content"]
    
    def extract_keywords(
        self,
        text: str,
        num_keywords: int = 5,
        temperature: float = 0.3
    ) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to extract
            temperature: Sampling temperature
            
        Returns:
            List of extracted keywords
        """
        messages = [
            {"role": "system", "content": f"Extract {num_keywords} keywords from the following text. Return only a comma-separated list of keywords without any additional text."},
            {"role": "user", "content": text}
        ]
        
        response = self.chat_completion(messages, temperature=temperature)
        
        # Parse response
        keywords_text = response["choices"][0]["message"]["content"]
        keywords = [kw.strip() for kw in keywords_text.split(",")]
        
        return keywords 