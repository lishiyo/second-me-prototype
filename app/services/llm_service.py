"""
LLMService for handling interactions with LLM models.

This module provides a service for interacting with LLM models,
handling retries, and providing utility functions for LLM-based generation.
"""
import logging
import os
import time
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI

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
        default_model: str = "gpt-4o-mini",
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
            
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Default topic parameters for LLM calls that need them
        # TODO: These are defaults for the topics generator, we should move them to the topics generator class
        self.topic_params = {
            "temperature": 0,
            "max_tokens": 1500,
            "top_p": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 5,
            "response_format": {"type": "json_object"},
        }
        self._top_p_adjusted = False  # Flag to track if top_p has been adjusted
    
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
    ) -> Any:
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
            API response object
        """
        model = model or self.default_model
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making OpenAI API call (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.client.chat.completions.create(
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
    
    def _fix_top_p_param(self, error_message: str) -> bool:
        """
        Fixes the top_p parameter if an API error indicates it's invalid.
        
        Some LLM providers don't accept top_p=0 and require values in specific ranges.
        This function checks if the error is related to top_p and adjusts it to 0.001,
        which is close enough to 0 to maintain deterministic behavior while satisfying
        API requirements.
        
        Args:
            error_message: Error message from the API response.
            
        Returns:
            bool: True if top_p was adjusted, False otherwise.
        """
        if not self._top_p_adjusted and "top_p" in error_message.lower():
            logger.warning("Fixing top_p parameter from 0 to 0.001 to comply with model API requirements")
            self.topic_params["top_p"] = 0.001
            self._top_p_adjusted = True
            return True
        return False
    
    def call_with_retry(self, messages: List[Dict[str, str]], model_params: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Calls the LLM API with automatic retry for parameter adjustments.
        
        This function handles making API calls to the language model while
        implementing automatic parameter fixes when errors occur. If the API
        rejects the call due to invalid top_p parameter, it will adjust the
        parameter value and retry the call once.
        
        Args:
            messages: List of messages for the API call.
            model_params: Custom model parameters to use instead of self.topic_params
            **kwargs: Additional parameters to pass to the API call.
            
        Returns:
            API response object from the language model.
            
        Raises:
            Exception: If the API call fails after all retries or for unrelated errors.
        """
        model = kwargs.pop('model', self.default_model)
        
        # Use provided model_params or fall back to self.topic_params
        params = model_params or self.topic_params
        
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                **params,
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API Error: {error_msg}")
            
            # Try to fix top_p parameter if needed
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                if self._fix_top_p_param(error_msg):
                    logger.info("Retrying LLM API call with adjusted top_p parameter")
                    # Use a copy of the params to avoid modifying the original
                    adjusted_params = params.copy()
                    adjusted_params["top_p"] = 0.001
                    return self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **adjusted_params,
                        **kwargs
                    )
            
            # Re-raise the exception
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
                
                response = self.client.embeddings.create(
                    model=model,
                    input=texts
                )
                
                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]
                
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
        
        return response.choices[0].message.content
    
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
        keywords_text = response.choices[0].message.content
        keywords = [kw.strip() for kw in keywords_text.split(",")]
        
        return keywords 