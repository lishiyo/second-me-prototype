from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Bio:
    """
    Represents a user biography with different perspective views.
    """
    # Identity fields
    id: str = ""
    user_id: str = ""
    version: int = 0
    
    # First person - "I am..."
    content_first_view: str = ""
    summary_first_view: str = ""
    
    # Second person - "You are..."
    content_second_view: str = ""
    summary_second_view: str = ""
    
    # Third person - "They are..."
    content_third_view: str = ""
    summary_third_view: str = ""
    
    # Health status assessment (physical and mental health)
    health_status: str = ""
    
    # Additional attributes
    confidence: float = 0.0
    shades_list: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "version": self.version,
            "content_first_view": self.content_first_view,
            "summary_first_view": self.summary_first_view,
            "content_second_view": self.content_second_view,
            "summary_second_view": self.summary_second_view,
            "content_third_view": self.content_third_view,
            "summary_third_view": self.summary_third_view,
            "health_status": self.health_status,
            "confidence": self.confidence,
            "shades_list": self.shades_list,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bio":
        """Create a Bio from a dictionary"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
            
        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            version=data.get("version", 0),
            content_first_view=data.get("content_first_view", ""),
            summary_first_view=data.get("summary_first_view", ""),
            content_second_view=data.get("content_second_view", ""),
            summary_second_view=data.get("summary_second_view", ""),
            content_third_view=data.get("content_third_view", ""),
            summary_third_view=data.get("summary_third_view", ""),
            health_status=data.get("health_status", ""),
            confidence=data.get("confidence", 0.0),
            shades_list=data.get("shades_list", []),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get("metadata", {})
        )
    
    def to_str(self) -> str:
        """Convert to string representation for prompting"""
        return f"""
Content (Third Person):
{self.content_third_view}

Summary (Third Person):
{self.summary_third_view}
"""
    
    def complete_content(self) -> str:
        """Get the complete content with summary"""
        return f"{self.summary_third_view}\n\n{self.content_third_view}"
    
    def shift_perspective_to_first(self) -> "Bio":
        """Shift perspective to first person"""
        # This would be implemented with LLM calls
        # For now, return self (placeholder)
        return self
    
    def shift_perspective_to_second(self) -> "Bio":
        """Shift perspective to second person"""
        # This would be implemented with LLM calls
        # For now, return self (placeholder)
        return self 