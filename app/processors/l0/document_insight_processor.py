from typing import Optional, Dict, Any

from app.processors.l0.models import (
    DocumentInsight, 
    DocumentSummary,
    FileInfo
)
from app.processors.l0.document_analyzer import (
    DocumentInsightGenerator,
    DocumentSummaryGenerator
)
from app.processors.l0.utils import setup_logger, retry

# Set up logger
logger = setup_logger(__name__)

class DocumentInsightProcessor:
    """
    Coordinates the two-stage document analysis process:
    1. Generate deep insights (DocumentInsightGenerator)
    2. Create concise summaries with keywords (DocumentSummaryGenerator)
    
    This follows the approach from lpm_kernel's L0 layer.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 insight_model: str = "gpt-3.5-turbo",
                 summary_model: str = "gpt-3.5-turbo",
                 insight_max_tokens: int = 1500,
                 summary_max_tokens: int = 1000):
        """
        Initialize the document insight processor.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            insight_model: OpenAI model to use for insight generation
            summary_model: OpenAI model to use for summary generation
            insight_max_tokens: Max tokens for insight generation
            summary_max_tokens: Max tokens for summary generation
        """
        self.insight_generator = DocumentInsightGenerator(
            api_key=api_key,
            model=insight_model,
            max_tokens=insight_max_tokens
        )
        
        self.summary_generator = DocumentSummaryGenerator(
            api_key=api_key,
            model=summary_model,
            max_tokens=summary_max_tokens
        )
        
        logger.info("DocumentInsightProcessor initialized")
    
    @retry(max_retries=1)
    def process_document(self, 
                        content: str, 
                        filename: str = "", 
                        document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document through the two-stage analysis pipeline:
        1. Generate insight
        2. Generate summary based on insight
        
        Args:
            content: Document content text
            filename: Original filename
            document_id: Optional document ID for logging
            
        Returns:
            Dictionary with insight and summary objects
        """
        doc_identifier = document_id or filename or "document"
        logger.info(f"Starting two-stage analysis for {doc_identifier}")
        
        # Stage 1: Generate deep insight
        logger.info(f"Stage 1: Generating insight for {doc_identifier}")
        insight = self.insight_generator.generate_insight(content, filename)
        logger.info(f"Insight generated for {doc_identifier}: {insight.title}")
        
        # Stage 2: Generate summary based on insight
        logger.info(f"Stage 2: Generating summary for {doc_identifier} based on insight")
        summary = self.summary_generator.generate_summary(content, insight, filename)
        logger.info(f"Summary generated for {doc_identifier}: {len(summary.keywords)} keywords")
        
        return {
            "insight": insight,
            "summary": summary
        }
    
    def process_from_file_info(self, file_info: FileInfo) -> Dict[str, Any]:
        """
        Process a document from FileInfo object.
        
        Args:
            file_info: FileInfo object containing document metadata and content
            
        Returns:
            Dictionary with insight and summary objects
        """
        return self.process_document(
            content=file_info.content,
            filename=file_info.filename,
            document_id=file_info.document_id
        ) 