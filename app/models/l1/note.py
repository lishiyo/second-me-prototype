from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime


@dataclass
class Chunk:
    """
    Represents a segment of a document with its own embedding.
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "document_id": self.document_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create a Chunk from a dictionary"""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            document_id=data.get("document_id"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Note:
    """
    Represents a processed document with metadata and embeddings.
    Similar to the Note class in lpm_kernel.
    """
    id: str
    content: str
    create_time: Union[datetime, str]
    embedding: Optional[List[float]] = None
    chunks: List[Chunk] = field(default_factory=list)
    title: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    insight: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    memory_type: str = "TEXT"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        # Handle create_time that could be a string or datetime object
        create_time_str = self.create_time
        if hasattr(self.create_time, 'isoformat'):
            create_time_str = self.create_time.isoformat()
        
        return {
            "id": self.id,
            "content": self.content,
            "create_time": create_time_str,
            "embedding": self.embedding,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "title": self.title,
            "summary": self.summary,
            "insight": self.insight,
            "tags": self.tags,
            "memory_type": self.memory_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Note":
        """Create a Note from a dictionary"""
        create_time = data.get("create_time")
        if isinstance(create_time, str):
            create_time = datetime.fromisoformat(create_time)
        
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            create_time=create_time or datetime.now(),
            embedding=data.get("embedding"),
            chunks=[Chunk.from_dict(chunk) for chunk in data.get("chunks", [])],
            title=data.get("title", ""),
            summary=data.get("summary", {}),
            insight=data.get("insight", {}),
            tags=data.get("tags", []),
            memory_type=data.get("memory_type", "TEXT"),
            metadata=data.get("metadata", {})
        ) 