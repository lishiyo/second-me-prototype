from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.models.l1.bio import Bio


@dataclass
class L1GenerationResult:
    """
    Encapsulates the complete result of the L1 generation process.
    """
    bio: Optional[Bio] = None
    clusters: Dict[str, Any] = field(default_factory=dict)
    chunk_topics: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    error: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "bio": self.bio.to_dict() if self.bio else None,
            "clusters": self.clusters,
            "chunk_topics": self.chunk_topics,
            "status": self.status,
            "error": self.error,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L1GenerationResult":
        """Create an L1GenerationResult from a dictionary"""
        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at)
            
        bio_data = data.get("bio")
        bio = Bio.from_dict(bio_data) if bio_data else None
            
        return cls(
            bio=bio,
            clusters=data.get("clusters", {}),
            chunk_topics=data.get("chunk_topics", []),
            status=data.get("status", "completed"),
            error=data.get("error"),
            generated_at=generated_at or datetime.now(),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def success(cls, bio: Bio, clusters: Dict[str, Any], chunk_topics: List[Dict[str, Any]]) -> "L1GenerationResult":
        """Create a successful generation result"""
        return cls(
            bio=bio,
            clusters=clusters,
            chunk_topics=chunk_topics,
            status="completed"
        )
    
    @classmethod
    def failure(cls, error: str) -> "L1GenerationResult":
        """Create a failed generation result"""
        return cls(
            status="failed",
            error=error
        ) 