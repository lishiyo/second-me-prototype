from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Topic:
    """
    Represents a thematic grouping of related documents.
    """
    id: str
    name: str
    summary: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "document_ids": self.document_ids,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Topic":
        """Create a Topic from a dictionary"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
            
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            summary=data.get("summary"),
            document_ids=data.get("document_ids", []),
            embedding=data.get("embedding"),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get("metadata", {})
        )


@dataclass
class Memory:
    """
    Represents a memory entry in a cluster, similar to lpm_kernel.
    Used for clustering documents by similarity.
    """
    memory_id: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "memory_id": self.memory_id,
            "embedding": self.embedding,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create a Memory from a dictionary"""
        return cls(
            memory_id=data.get("memory_id", ""),
            embedding=data.get("embedding", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class Cluster:
    """
    Represents a group of related documents clustered by similarity.
    """
    id: str
    topic_id: Optional[str] = None
    name: Optional[str] = None
    summary: Optional[str] = None
    memory_list: List[Memory] = field(default_factory=list)
    center_embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def document_ids(self) -> List[str]:
        """Get list of document IDs in this cluster"""
        return [memory.memory_id for memory in self.memory_list]
    
    @property
    def document_count(self) -> int:
        """Get count of documents in this cluster"""
        return len(self.memory_list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "topic_id": self.topic_id,
            "name": self.name,
            "summary": self.summary,
            "memory_list": [memory.to_dict() for memory in self.memory_list],
            "center_embedding": self.center_embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cluster":
        """Create a Cluster from a dictionary"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
            
        return cls(
            id=data.get("id", ""),
            topic_id=data.get("topic_id"),
            name=data.get("name"),
            summary=data.get("summary"),
            memory_list=[Memory.from_dict(m) for m in data.get("memory_list", [])],
            center_embedding=data.get("center_embedding"),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get("metadata", {})
        ) 