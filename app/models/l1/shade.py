from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ShadeInfo:
    """
    Contains information about existing shades for a cluster.
    Used when generating/updating shades.
    """
    shade_id: str
    name: str
    content: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "shade_id": self.shade_id,
            "name": self.name,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadeInfo":
        """Create a ShadeInfo from a dictionary"""
        return cls(
            shade_id=data.get("shade_id", ""),
            name=data.get("name", ""),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class Shade:
    """
    Represents a knowledge aspect extracted from document clusters.
    """
    id: str
    name: str
    user_id: str = ""
    summary: Optional[str] = None
    content: str = ""
    confidence: float = 0.0
    source_clusters: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    s3_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "summary": self.summary,
            "content": self.content,
            "confidence": self.confidence,
            "source_clusters": self.source_clusters,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "s3_path": self.s3_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Shade":
        """Create a Shade from a dictionary"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
            
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            user_id=data.get("user_id", ""),
            summary=data.get("summary"),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.0),
            source_clusters=data.get("source_clusters", []),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get("metadata", {}),
            s3_path=data.get("s3_path")
        )


@dataclass
class ShadeMergeInfo:
    """
    Information about shades to be merged.
    """
    shade_id: str
    name: str
    summary: Optional[str] = None
    content: str = ""
    confidence: float = 0.0
    source_clusters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "shade_id": self.shade_id,
            "name": self.name,
            "summary": self.summary,
            "content": self.content,
            "confidence": self.confidence,
            "source_clusters": self.source_clusters,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadeMergeInfo":
        """Create a ShadeMergeInfo from a dictionary"""
        return cls(
            shade_id=data.get("shade_id", ""),
            name=data.get("name", ""),
            summary=data.get("summary"),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.0),
            source_clusters=data.get("source_clusters", []),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_shade(cls, shade: Shade) -> "ShadeMergeInfo":
        """Create a ShadeMergeInfo from a Shade"""
        return cls(
            shade_id=shade.id,
            name=shade.name,
            summary=shade.summary,
            content=shade.content,
            confidence=shade.confidence,
            source_clusters=shade.source_clusters,
            metadata=shade.metadata
        )


@dataclass
class MergedShadeResult:
    """
    Result of a shade merging operation.
    """
    success: bool
    merge_shade_list: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "success": self.success,
            "merge_shade_list": self.merge_shade_list,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergedShadeResult":
        """Create a MergedShadeResult from a dictionary"""
        return cls(
            success=data.get("success", False),
            merge_shade_list=data.get("merge_shade_list", []),
            error=data.get("error")
        ) 