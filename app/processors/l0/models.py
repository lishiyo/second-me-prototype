from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingStatus(str, Enum):
    """
    Enum for document processing status.
    """
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class FileInfo:
    """
    Class to represent document metadata.
    """
    document_id: str
    filename: str
    content_type: str
    s3_path: str
    content: Optional[str] = None


@dataclass
class ChunkInfo:
    """
    Class to represent document chunk information.
    """
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    s3_path: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    

@dataclass
class BioInfo:
    """
    User biographical information.
    """
    global_bio: str = ""
    status_bio: str = ""
    about_me: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BioInfo":
        """
        Create a BioInfo instance from a dictionary.
        
        Args:
            data: Dictionary containing bio information
            
        Returns:
            A BioInfo instance
        """
        return cls(
            global_bio=data.get("global_bio", ""),
            status_bio=data.get("status_bio", ""),
            about_me=data.get("about_me", "")
        )


@dataclass
class InsighterInput:
    """
    Raw input parameters for the Insighter component.
    Used for document analysis and insight generation.
    """
    file_info: FileInfo
    bio_info: BioInfo

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> "InsighterInput":
        """
        Creates an InsighterInput instance from a dictionary.
        
        Args:
            inputs: Dictionary containing the input parameters.
            
        Returns:
            An InsighterInput object populated with values from the dictionary.
        """
        file_info = FileInfo(
            document_id=inputs.get("document_id", ""),
            filename=inputs.get("filename", ""),
            content_type=inputs.get("content_type", ""),
            s3_path=inputs.get("s3_path", ""),
            content=inputs.get("content", ""),
        )
        
        bio_info_dict = inputs.get("bio_info", {})
        bio_info = BioInfo.from_dict(bio_info_dict) if isinstance(bio_info_dict, dict) else BioInfo()
        
        return cls(
            file_info=file_info,
            bio_info=bio_info,
        )


@dataclass
class SummarizerInput:
    """
    Raw input parameters for the Summarizer component.
    Used for generating summaries of documents with insights.
    """
    file_info: FileInfo
    insight: str

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> "SummarizerInput":
        """
        Creates a SummarizerInput instance from a dictionary.
        
        Args:
            inputs: Dictionary containing the input parameters.
            
        Returns:
            A SummarizerInput object populated with values from the dictionary.
        """
        file_info = FileInfo(
            document_id=inputs.get("document_id", ""),
            filename=inputs.get("filename", ""),
            content_type=inputs.get("content_type", ""),
            s3_path=inputs.get("s3_path", ""),
            content=inputs.get("content", ""),
        )
        
        return cls(
            file_info=file_info,
            insight=inputs.get("insight", ""),
        )


@dataclass
class DocumentType(str, Enum):
    """Document type classification."""
    DOCUMENT = "DOCUMENT"
    TEXT = "TEXT"
    
    @classmethod
    def from_mime_type(cls, mime_type: str) -> "DocumentType":
        """Converts a MIME type to DocumentType.
        
        Args:
            mime_type: String representing the MIME type.
            
        Returns:
            The appropriate DocumentType based on the given MIME type.
        """
        if mime_type == "text":
            return cls.TEXT
        elif mime_type == "pdf":
            return cls.DOCUMENT
        elif mime_type == "md":
            return cls.DOCUMENT
        else:
            return cls.DOCUMENT


@dataclass
class DocumentInsight:
    """
    Class to represent document deep insight analysis results.
    This is the first stage of document analysis, providing a detailed understanding.
    """
    title: str
    insight: str


@dataclass
class DocumentSummary:
    """
    Class to represent document summary results.
    This is the second stage of document analysis, building on the insights
    to provide a more concise representation with keywords.
    """
    title: str
    summary: str
    keywords: List[str]


@dataclass
class ProcessingResult:
    """
    Class to represent the processing result for a document.
    """
    document_id: str
    status: ProcessingStatus
    message: Optional[str] = None
    chunk_count: int = 0
    error: Optional[str] = None
    insight: Optional[DocumentInsight] = None
    summary: Optional[DocumentSummary] = None
    
    @classmethod
    def success(cls, 
                document_id: str, 
                chunk_count: int, 
                insight: Optional[DocumentInsight] = None,
                summary: Optional[DocumentSummary] = None) -> "ProcessingResult":
        """Create a successful processing result."""
        return cls(
            document_id=document_id,
            status=ProcessingStatus.COMPLETED,
            message="Processing completed successfully",
            chunk_count=chunk_count,
            insight=insight,
            summary=summary
        )
    
    @classmethod
    def failure(cls, document_id: str, error: str) -> "ProcessingResult":
        """Create a failed processing result."""
        return cls(
            document_id=document_id,
            status=ProcessingStatus.FAILED,
            message=f"Processing failed: {error}",
            error=error
        )
    
    @classmethod
    def in_progress(cls, document_id: str) -> "ProcessingResult":
        """Create an in-progress processing result."""
        return cls(
            document_id=document_id,
            status=ProcessingStatus.PROCESSING,
            message="Processing in progress"
        )
