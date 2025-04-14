import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import openai
from openai import OpenAI

from app.processors.l0.models import DocumentInsight, DocumentSummary
from app.processors.l0.utils import retry, setup_logger
from app.processors.l0.prompts import DOCUMENT_INSIGHT_PROMPT, DOCUMENT_SUMMARY_PROMPT, DOCUMENT_TITLE_SUMMARY_PROMPT

# Set up logger
logger = setup_logger(__name__)

class DocumentInsightGenerator:
    """
    Generates deep insights from document content.
    This is the first stage of document analysis, providing detailed understanding.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 1500):
        """
        Initialize the document insight generator.
        
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
    def generate_insight(self, content: str, filename: str = "") -> DocumentInsight:
        """
        Generate deep insight from document content.
        
        Args:
            content: Document content text
            filename: Original filename (optional)
            
        Returns:
            DocumentInsight object with title and detailed insight
        """
        logger.info(f"Generating deep insight for document: {filename}")
        
        # Truncate content if it's too long to avoid token limits
        truncated_content = self._truncate_content(content, max_length=5000)
        
        # Generate title and insight using OpenAI
        try:
            title, insight = self._generate_title_and_insight(truncated_content, filename)
        except Exception as e:
            logger.error(f"Error generating title and insight: {str(e)}")
            # Use fallback title and a simple error message
            title = self._fallback_title(truncated_content, filename)
            insight = f"Failed to generate insight due to API error: {str(e)}"
        
        logger.info(f"Insight generation complete - Title: {title[:30]}...")
        return DocumentInsight(
            title=title,
            insight=insight
        )
    
    def _truncate_content(self, content: str, max_length: int = 5000) -> str:
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
    
    def _generate_title_and_insight(self, content: str, filename: str = "") -> Tuple[str, str]:
        """
        Generate a title and detailed insight for document content using OpenAI.
        
        Args:
            content: Document content text
            filename: Original filename
            
        Returns:
            Tuple of (title, insight)
        """
        # Include filename in the prompt if available
        filename_info = f"Filename: {filename}\n\n" if filename else ""
        
        prompt = f"""
        {filename_info}Document content:
        {content}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DOCUMENT_INSIGHT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # Parse the result assuming it's JSON
        result_json = json.loads(result)
        
        title = result_json.get("title", self._fallback_title(content, filename))
        
        # Get the insight, which could be a string or a nested object
        insight_data = result_json.get("insight", {})
        if isinstance(insight_data, str):
            insight = insight_data
        else:
            # Format the structured insight into a string
            overview = insight_data.get("overview", "")
            breakdown = insight_data.get("breakdown", {})
            
            formatted_insight = overview + "\n\n"
            
            for section_title, conclusions in breakdown.items():
                formatted_insight += f"{section_title}\n"
                for conclusion in conclusions:
                    if len(conclusion) >= 2:
                        key_point, explanation = conclusion[0], conclusion[1]
                        formatted_insight += f"• {key_point}: {explanation}\n"
                formatted_insight += "\n"
            
            insight = formatted_insight.strip()
        
        if not insight:
            insight = "No insight available."
        
        return title, insight
    
    def _fallback_title(self, content: str, filename: str = "") -> str:
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
                # Clean up markdown heading symbols
                line = re.sub(r'^#+\s+', '', line)
                return line
        
        # Use first sentence if it's short enough
        first_sentence_match = re.search(r'^(.+?[.!?])\s', content)
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1)
            # Clean up markdown heading symbols
            first_sentence = re.sub(r'^#+\s+', '', first_sentence)
            if len(first_sentence) < 100:
                return first_sentence
        
        # Fallback to filename or generic title
        if filename:
            # Remove extension and replace underscores/hyphens with spaces
            base_name = os.path.splitext(os.path.basename(filename))[0]
            title = re.sub(r'[_-]', ' ', base_name).title()
            return title
        
        return "Untitled Document"


class DocumentSummaryGenerator:
    """
    Generates concise summaries and keywords from document content and insights.
    This is the second stage of document analysis, building on the insights.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000):
        """
        Initialize the document summary generator.
        
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
    def generate_summary(self, content: str, insight: DocumentInsight, filename: str = "") -> DocumentSummary:
        """
        Generate a concise summary and keywords based on content and previous insight.
        
        Args:
            content: Document content text
            insight: Previously generated document insight
            filename: Original filename (optional)
            
        Returns:
            DocumentSummary object with title, summary, and keywords
        """
        logger.info(f"Generating summary and keywords based on insight for document: {filename}")
        
        # Use the same title from the insight
        title = insight.title
        
        # Generate summary and keywords using OpenAI
        try:
            summary, keywords = self._generate_summary_and_keywords(content, insight.insight, title)
        except Exception as e:
            logger.error(f"Error generating summary and keywords: {str(e)}")
            # Default fallback summary
            summary = "Failed to generate summary due to API error."
            keywords = ["document"]
        
        logger.info(f"Summary generation complete - Title: {title[:30]}...")
        return DocumentSummary(
            title=title,
            summary=summary,
            keywords=keywords
        )
    
    def _generate_summary_and_keywords(self, content: str, insight: str, title: str) -> Tuple[str, List[str]]:
        """
        Generate summary and keywords based on content and previous insight using OpenAI.
        
        Args:
            content: Document content text
            insight: Previously generated insight
            title: Document title
            
        Returns:
            Tuple of (summary, keywords list)
        """
        prompt = f"""
        Document title: {title}
        
        Previous insight:
        {insight}
        
        Document content sample (first 1000 chars):
        {content[:1000]}...
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DOCUMENT_SUMMARY_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # Parse the result assuming it's JSON
        try:
            result_json = json.loads(result)
            summary = result_json.get("summary", "No summary available.")
            keywords = result_json.get("keywords", ["document"])
            
            # Ensure keywords is a list
            if not isinstance(keywords, list):
                keywords = [keywords]
                
            # Limit number of keywords if needed
            keywords = keywords[:10]
            
            return summary, keywords
        except json.JSONDecodeError:
            logger.error("Failed to parse summary response as JSON")
            return "No summary available.", ["document"]

    def generate_title_and_summary(self, content: str, filename: str = "") -> Dict[str, Any]:
        """
        Generate title, summary and keywords in a single call (alternative approach).
        This is similar to the summarizer method in lpm_kernel.
        
        Args:
            content: Document content text
            filename: Original filename (optional)
            
        Returns:
            Dictionary with title, summary, and keywords
        """
        # Limit content length
        if len(content) > 8000:
            content = content[:8000] + "..."
        
        # Prepare filename description
        filename_desc = f"Filename: {filename}" if filename else ""
        
        # Format the prompt with content and filename
        # Use string replacement instead of format() to avoid issues with curly braces in the prompt
        prompt = DOCUMENT_TITLE_SUMMARY_PROMPT.replace("{filename_desc}", filename_desc).replace("{content}", content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document analyzer that extracts the essential information from documents. Provide your analysis in JSON format with 'title', 'summary', and 'keywords' fields."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # Parse the result as JSON
            result_json = json.loads(result)
            
            title = result_json.get("title", self._fallback_title(content, filename))
            summary = result_json.get("summary", "No summary available.")
            keywords = result_json.get("keywords", ["document"])
            
            # Ensure keywords is a list
            if not isinstance(keywords, list):
                keywords = [keywords]
                
            return {
                "title": title,
                "summary": summary,
                "keywords": keywords
            }
            
        except Exception as e:
            logger.error(f"Error generating title and summary: {str(e)}")
            return {
                "title": self._fallback_title(content, filename),
                "summary": "Failed to generate summary due to API error.",
                "keywords": ["document"]
            }

    def _fallback_title(self, content: str, filename: str = "") -> str:
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
                # Clean up markdown heading symbols
                line = re.sub(r'^#+\s+', '', line)
                return line
        
        # Fallback to filename or generic title
        if filename:
            # Remove extension and replace underscores/hyphens with spaces
            base_name = os.path.splitext(os.path.basename(filename))[0]
            title = re.sub(r'[_-]', ' ', base_name).title()
            return title
            
        return "Untitled Document"


# For backward compatibility, alias DocumentSummaryGenerator as DocumentAnalyzer
DocumentAnalyzer = DocumentSummaryGenerator
