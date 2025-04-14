from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.models.l1.bio import Bio


@dataclass
class L1GenerationResult:
    """
    Encapsulates the complete result of the L1 generation process.
    Compatible with lpm_kernel's L1GenerationResult.
    """
    bio: Optional[Bio] = None
    clusters: Dict[str, List] = field(default_factory=lambda: {"clusterList": []})  # {"clusterList": [...]}
    chunk_topics: Dict[str, Dict] = field(default_factory=dict)  # {cluster_id: {"indices": [], "docIds": [], ...}}
    generate_time: datetime = field(default_factory=datetime.now)  # Renamed from generated_at to match lpm_kernel
    status: str = "completed"  # Our improvement
    error: Optional[str] = None  # Our improvement
    metadata: Dict[str, Any] = field(default_factory=dict)  # Our improvement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            # Match lpm_kernel's structure while enhancing with our improvements
            "bio": self.bio.to_dict() if self.bio else None,  # Use to_dict for proper serialization
            "clusters": self.clusters,
            "chunk_topics": self.chunk_topics,
            "generate_time": self.generate_time.isoformat(),  # Renamed from generated_at
            # Our additional fields
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L1GenerationResult":
        """Create an L1GenerationResult from a dictionary"""
        generate_time = data.get("generate_time")  # Updated field name
        if isinstance(generate_time, str):
            generate_time = datetime.fromisoformat(generate_time)
            
        bio_data = data.get("bio")
        bio = Bio.from_dict(bio_data) if bio_data else None
            
        return cls(
            bio=bio,
            clusters=data.get("clusters", {"clusterList": []}),
            chunk_topics=data.get("chunk_topics", {}),
            generate_time=generate_time or datetime.now(),  # Updated field name
            status=data.get("status", "completed"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def success(cls, bio: Bio, clusters: Dict[str, List], chunk_topics: Dict[str, Dict]) -> "L1GenerationResult":
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
            clusters={"clusterList": []},  # Initialize with expected structure
            chunk_topics={},
            status="failed",
            error=error
        ) 