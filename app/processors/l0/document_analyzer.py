import os
import re
from typing import List, Dict, Any, Optional, Tuple

import openai
from openai import OpenAI

from app.processors.l0.models import DocumentInsights
from app.processors.l0.utils import retry, setup_logger

# Set up logger
logger = setup_logger(__name__)

class DocumentAnalyzer:
    """
    Analyzes document content to extract insights like title, summary, and keywords.
    Uses OpenAI's models for text analysis.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000):
        """
        Initialize the document analyzer.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for analysis
            max_tokens: Maximum number of tokens in the response
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass to constructor.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    @retry(max_retries=2, error_types=(openai.RateLimitError, openai.APITimeoutError))
    def analyze(self, content: str, filename: str = "") -> DocumentInsights:
        """
        Analyze document content to extract insights.
        
        Args:
            content: Document content text
            filename: Original filename (optional)
            
        Returns:
            DocumentInsights object with title, summary, and keywords
        """
        logger.info(f"Analyzing document content for insights")
        
        # Truncate content if it's too long to avoid token limits
        truncated_content = self._truncate_content(content, max_length=4000)
        
        # Detect title from content or use filename
        title = self._extract_title(truncated_content, filename)
        
        # Generate summary and keywords using OpenAI
        summary, keywords = self._generate_summary_and_keywords(truncated_content)
        
        logger.info(f"Analysis complete - Title: {title[:30]}...")
        return DocumentInsights(
            title=title,
            summary=summary,
            keywords=keywords
        )
    
    def _truncate_content(self, content: str, max_length: int = 4000) -> str:
        """
        Truncate content to a maximum length while preserving coherent paragraphs.
        
        Args:
            content: Document content text
            max_length: Maximum characters to include
            
        Returns:
            Truncated content
        """
        if len(content) <= max_length:
            return content
        
        # Try to truncate at paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', content)
        truncated = ""
        
        for paragraph in paragraphs:
            if len(truncated + paragraph) > max_length:
                break
            truncated += paragraph + "\n\n"
        
        # If we couldn't get any complete paragraphs, just truncate at max_length
        if not truncated:
            truncated = content[:max_length]
        
        return truncated.strip() + "..."
    
    def _extract_title(self, content: str, filename: str = "") -> str:
        """
        Extract title from document content or fallback to filename.
        
        Args:
            content: Document content text
            filename: Original filename
            
        Returns:
            Extracted title
        """
        # Try to find a title in the first few lines of the document
        lines = content.split('\n')
        for i in range(min(5, len(lines))):
            line = lines[i].strip()
            # Look for lines that might be titles (short, no punctuation at end)
            if line and len(line) < 100 and not line.endswith(('.', ':', ';', '!', '?')):
                return line
        
        # Use first sentence if it's short enough
        first_sentence_match = re.search(r'^(.+?[.!?])\s', content)
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1)
            if len(first_sentence) < 100:
                return first_sentence
        
        # Fallback to filename or generic title
        if filename:
            # Remove extension and replace underscores/hyphens with spaces
            base_name = os.path.splitext(os.path.basename(filename))[0]
            title = re.sub(r'[_-]', ' ', base_name).title()
            return title
        
        return "Untitled Document"
    
    def _generate_summary_and_keywords(self, content: str) -> Tuple[str, List[str]]:
        """
        Generate summary and keywords for document content using OpenAI.
        
        Args:
            content: Document content text
            
        Returns:
            Tuple of (summary, keywords list)
        """
        prompt = f"""
        Please analyze this document content and provide:
        1. A concise summary (3-5 sentences)
        2. Up to 10 relevant keywords or key phrases

        Format your response as JSON with 'summary' and 'keywords' fields.
        
        Document content:
        {content}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document analysis assistant that extracts summaries and keywords from documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # Parse the result assuming it's JSON
            import json
            result_json = json.loads(result)
            
            summary = result_json.get("summary", "No summary available.")
            keywords = result_json.get("keywords", [])
            
            # Ensure keywords are a list
            if isinstance(keywords, str):
                # If keywords are a comma-separated string, split them
                keywords = [k.strip() for k in keywords.split(",")]
            
            return summary, keywords
            
        except Exception as e:
            logger.error(f"Error generating summary and keywords: {str(e)}")
            # Return default values if API call fails
            return "No summary available.", ["document"]
