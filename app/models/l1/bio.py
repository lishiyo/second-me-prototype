from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Bio:
    """
    Represents a user biography with different perspective views.
    Similar to the Bio class in lpm_kernel.
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
    attribute_list: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Properties for compatibility with lpm_kernel Bio class
    @property
    def contentThirdView(self) -> str:
        """Compatibility with lpm_kernel: alias for content_third_view"""
        return self.content_third_view
        
    @contentThirdView.setter
    def contentThirdView(self, value: str):
        """Compatibility with lpm_kernel: alias for content_third_view"""
        self.content_third_view = value
        
    @property
    def content(self) -> str:
        """Compatibility with lpm_kernel: alias for content_second_view"""
        return self.content_second_view
        
    @content.setter
    def content(self, value: str):
        """Compatibility with lpm_kernel: alias for content_second_view"""
        self.content_second_view = value
        
    @property
    def summaryThirdView(self) -> str:
        """Compatibility with lpm_kernel: alias for summary_third_view"""
        return self.summary_third_view
        
    @summaryThirdView.setter
    def summaryThirdView(self, value: str):
        """Compatibility with lpm_kernel: alias for summary_third_view"""
        self.summary_third_view = value
        
    @property
    def summary(self) -> str:
        """Compatibility with lpm_kernel: alias for summary_second_view"""
        return self.summary_second_view
        
    @summary.setter
    def summary(self, value: str):
        """Compatibility with lpm_kernel: alias for summary_second_view"""
        self.summary_second_view = value
        
    @property
    def attributeList(self) -> List[Dict[str, Any]]:
        """Compatibility with lpm_kernel: alias for attribute_list"""
        return self.attribute_list
        
    @attributeList.setter
    def attributeList(self, value: List[Dict[str, Any]]):
        """Compatibility with lpm_kernel: alias for attribute_list"""
        self.attribute_list = value
        
    @property
    def shadesList(self) -> List[Dict[str, Any]]:
        """Compatibility with lpm_kernel: alias for shades_list"""
        return self.shades_list
        
    @shadesList.setter
    def shadesList(self, value: List[Dict[str, Any]]):
        """Compatibility with lpm_kernel: alias for shades_list"""
        self.shades_list = value
    
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
            "attribute_list": self.attribute_list,
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
            attribute_list=data.get("attribute_list", []),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get("metadata", {})
        )
    
    def to_str(self) -> str:
        """Convert to string representation for prompting"""
        global_bio_statement = ""
        if self.is_raw_bio():
            global_bio_statement += f"**[Origin Analysis]**\n{self.summary_third_view}\n"
        
        global_bio_statement += f"\n**[Current Shades]**\n"
        for i, shade in enumerate(self.shades_list):
            # Format each shade similar to how ShadeInfo.to_str() would do it
            # Handle both dictionary access and object attribute access
            if isinstance(shade, dict):
                name = shade.get('name', '')
                summary = shade.get('summary', '')
                timelines = shade.get('timelines', [])
            else:
                # Handle object attribute access
                name = getattr(shade, 'name', '')
                summary = getattr(shade, 'summary', '')
                timelines = getattr(shade, 'timelines', [])
            
            shade_str = f"**[Name]**: {name}\n"
            shade_str += f"**[Description]**: {summary}\n"
            
            # Add timeline information if available
            if timelines:
                shade_str += "**[Timelines]**:\n"
                for timeline in timelines:
                    if isinstance(timeline, dict):
                        created_at = timeline.get('created_at', '')
                        description = timeline.get('description', '')
                    else:
                        created_at = getattr(timeline, 'created_at', '')
                        description = getattr(timeline, 'description', '')
                    
                    shade_str += f"- {created_at}, {description}\n"
            
            global_bio_statement += shade_str
            global_bio_statement += "\n==============\n"
        
        return global_bio_statement
    
    def complete_content(self, second_view: bool = False) -> str:
        """
        Get the complete content with summary.
        
        Args:
            second_view: Whether to use second-person perspective content and summary
            
        Returns:
            Complete formatted content
        """
        interests_preference_field = (
            "\n### User's Interests and Preferences ###\n"
        )
        
        # Add preview of each shade
        for shade in self.shades_list:
            # Handle both dictionary access and object attribute access
            if isinstance(shade, dict):
                name = shade.get('name', '')
                summary = shade.get('summary', '')
            else:
                name = getattr(shade, 'name', '')
                summary = getattr(shade, 'summary', '')
                
            interests_preference_field += f"- {name}: {summary}\n"
        
        # Use either third-person or second-person summary based on the second_view parameter
        if not second_view:
            conclusion_field = "\n### Conclusion ###\n" + self.summary_third_view
        else:
            conclusion_field = "\n### Conclusion ###\n" + self.summary_second_view
        
        return f"""## Comprehensive Analysis Report ##
{interests_preference_field}
{conclusion_field}"""
    
    # Add methods for compatibility with lpm_kernel
    def is_raw_bio(self) -> bool:
        """Check if this is a raw bio (minimal structure)"""
        return not bool(self.content_second_view or self.content_third_view)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format for compatibility with lpm_kernel"""
        return {
            "contentThirdView": self.content_third_view,
            "content": self.content_second_view,  
            "summaryThirdView": self.summary_third_view,
            "summary": self.summary_second_view,
            "attributeList": self.attribute_list,
            "shadesList": self.shades_list
        }
    
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