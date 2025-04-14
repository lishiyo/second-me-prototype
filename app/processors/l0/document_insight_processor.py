from typing import Optional, Dict, Any

from app.processors.l0.models import (
    DocumentInsight, 
    DocumentSummary,
    FileInfo,
    BioInfo,
    InsighterInput
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
        
    def process_with_bio(self, file_info: FileInfo, bio_info: BioInfo) -> Dict[str, Any]:
        """
        Process a document with biographical information, similar to _insighter_doc in lpm_kernel.
        This can provide more personalized insights based on user information.
        
        Args:
            file_info: FileInfo object containing document metadata and content
            bio_info: BioInfo object containing user biographical information
            
        Returns:
            Dictionary with insight and summary objects
        """
        logger.info(f"Processing document with biographical context: {file_info.filename}")
        
        # First, generate the basic insight using standard processing
        result = self.process_document(
            content=file_info.content,
            filename=file_info.filename,
            document_id=file_info.document_id
        )
        
        insight = result["insight"]
        
        # Add biographical context to the insight
        insight.insight = self._add_bio_context_to_insight(insight.insight, bio_info)
        
        # Re-generate the summary with the bio-enhanced insight
        summary = self.summary_generator.generate_summary(file_info.content, insight, file_info.filename)
        
        return {
            "insight": insight,
            "summary": summary
        }
    
    def _add_bio_context_to_insight(self, insight: str, bio_info: BioInfo) -> str:
        """
        Add biographical context to an insight.
        
        Args:
            insight: Original document insight
            bio_info: User biographical information
            
        Returns:
            Enhanced insight with biographical context
        """
        # Only add biographical context if we have some bio information
        if not (bio_info.about_me or bio_info.global_bio or bio_info.status_bio):
            return insight
            
        # Get the biographical context
        bio_context = self._generate_bio_context(insight, bio_info)
        
        # Add the biographical context to the insight
        enhanced_insight = insight + "\n\n" + bio_context
        
        return enhanced_insight
    
    def _generate_bio_context(self, insight: str, bio_info: BioInfo) -> str:
        """
        Generate biographical context for an insight.
        
        Args:
            insight: Original document insight
            bio_info: User biographical information
            
        Returns:
            Biographical context
        """
        try:
            # Use the insight generator's client to generate bio context
            prompt = f"""
            Based on the following user information:
            
            User self-description: {bio_info.about_me}
            User's interests and background: {bio_info.global_bio}
            User's recent activities: {bio_info.status_bio}
            
            And the following document insight:
            
            {insight}
            
            Please generate a brief paragraph (3-5 sentences) that connects the document content to the user's 
            personal context, interests, or recent activities. Make the connection feel natural and relevant.
            """
            
            response = self.insight_generator.client.chat.completions.create(
                model=self.insight_generator.model,
                messages=[
                    {"role": "system", "content": "You are an assistant that helps connect document content to a user's personal context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.4
            )
            
            bio_context = "Personal Context: " + response.choices[0].message.content.strip()
            return bio_context
            
        except Exception as e:
            logger.error(f"Error generating biographical context: {str(e)}")
            return "Personal Context: Unable to generate personalized context."
    
    def process_from_insighter_input(self, input_data: InsighterInput) -> Dict[str, Any]:
        """
        Process from an InsighterInput object, similar to how lpm_kernel processes.
        
        Args:
            input_data: InsighterInput object containing file and bio information
            
        Returns:
            Dictionary with insight and summary objects
        """
        return self.process_with_bio(
            file_info=input_data.file_info,
            bio_info=input_data.bio_info
        ) 