from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json


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
    
    # Add LPM Kernel compatible fields
    aspect: str = ""
    icon: str = ""
    desc_third_view: str = ""
    content_third_view: str = ""
    desc_second_view: str = ""
    content_second_view: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "shade_id": self.shade_id,
            "name": self.name,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "aspect": self.aspect,
            "icon": self.icon,
            "desc_third_view": self.desc_third_view,
            "content_third_view": self.content_third_view,
            "desc_second_view": self.desc_second_view,
            "content_second_view": self.content_second_view
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadeInfo":
        """Create a ShadeInfo from a dictionary"""
        return cls(
            shade_id=data.get("shade_id", ""),
            name=data.get("name", ""),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata", {}),
            aspect=data.get("aspect", ""),
            icon=data.get("icon", ""),
            desc_third_view=data.get("desc_third_view", ""),
            content_third_view=data.get("content_third_view", ""),
            desc_second_view=data.get("desc_second_view", ""),
            content_second_view=data.get("content_second_view", "")
        )
    
    @classmethod
    def from_l1_shade(cls, shade: "L1Shade") -> "ShadeInfo":
        """Create a ShadeInfo from an L1Shade object"""
        return cls(
            shade_id=shade.id,
            name=shade.name,
            content=shade.summary,  # Map summary to content for backwards compatibility
            confidence=shade.confidence,
            metadata=shade.metadata,
            aspect=shade.aspect,
            icon=shade.icon,
            desc_third_view=shade.desc_third_view,
            content_third_view=shade.content_third_view,
            desc_second_view=shade.desc_second_view,
            content_second_view=shade.content_second_view
        )
    
    def to_l1_shade(self, user_id: str = None) -> "L1Shade":
        """Convert to an L1Shade object"""
        return L1Shade(
            id=self.shade_id,
            user_id=user_id,
            name=self.name,
            summary=self.content,  # Map content to summary for backwards compatibility
            confidence=self.confidence,
            metadata=self.metadata,
            aspect=self.aspect,
            icon=self.icon,
            desc_third_view=self.desc_third_view,
            content_third_view=self.content_third_view or self.content,
            desc_second_view=self.desc_second_view,
            content_second_view=self.content_second_view
        )

@dataclass
class ShadeMergeInfo:
    """
    Information about shades to be merged.
    Similar to lpm_kernel's ShadeMergeInfo but as a dataclass.
    """
    shade_id: str
    name: str
    aspect: str = ""
    icon: str = ""
    desc_third_view: str = ""
    content_third_view: str = ""
    desc_second_view: str = ""
    content_second_view: str = ""
    cluster_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "shade_id": self.shade_id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "desc_third_view": self.desc_third_view,
            "content_third_view": self.content_third_view,
            "desc_second_view": self.desc_second_view,
            "content_second_view": self.content_second_view,
            "cluster_info": self.cluster_info,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadeMergeInfo":
        """Create a ShadeMergeInfo from a dictionary"""
        return cls(
            shade_id=data.get("shade_id", ""),
            name=data.get("name", ""),
            aspect=data.get("aspect", ""),
            icon=data.get("icon", ""),
            desc_third_view=data.get("desc_third_view", ""),
            content_third_view=data.get("content_third_view", ""),
            desc_second_view=data.get("desc_second_view", ""),
            content_second_view=data.get("content_second_view", ""),
            cluster_info=data.get("cluster_info"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_shade(cls, shade: "L1Shade") -> "ShadeMergeInfo":
        """Create a ShadeMergeInfo from an L1Shade"""
        return cls(
            shade_id=shade.id,
            name=shade.name,
            aspect=shade.aspect,
            icon=shade.icon,
            desc_third_view=shade.desc_third_view,
            content_third_view=shade.content_third_view,
            desc_second_view=shade.desc_second_view,
            content_second_view=shade.content_second_view,
            metadata=shade.metadata
        )
        
    def improve_shade_info(self, improved_desc: str, improved_content: str) -> None:
        """Improve the shade with new description and content"""
        self.desc_third_view = improved_desc
        self.content_third_view = improved_content
        
    def add_second_view(self, domain_desc: str, domain_content: str) -> None:
        """Add second-person view information"""
        self.desc_second_view = domain_desc
        self.content_second_view = domain_content
        
    def to_str(self) -> str:
        """Format the shade as a string"""
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Aspect]**: {self.aspect}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.desc_third_view}\n\n**[Content]**: \n{self.content_third_view}\n"
        shade_statement += "---\n\n"
        if self.cluster_info:
            shade_statement += f"**[Cluster Info]**: \n{json.dumps(self.cluster_info, indent=2)}\n"
        return shade_statement


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


class L1Shade:
    """
    Data model for a shade (knowledge aspect).
    
    This class represents a shade, which is a specific aspect of knowledge
    extracted from document clusters. This is the same as lpm_kernel's ShadeInfo.
    
    Attributes:
        id: Unique identifier
        user_id: ID of the user this shade belongs to
        name: Name of the shade
        summary: Short summary of the shade
        confidence: Confidence score (0-1)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
        s3_path: Path to detailed data in storage
        aspect: The aspect/category of the shade
        icon: Icon reference for the shade
        desc_second_view: Description in second person
        desc_third_view: Description in third person
        content_second_view: Content in second person
        content_third_view: Content in third person
    """
    
    def __init__(
        self,
        id: str = None,
        user_id: str = None,
        name: str = "",
        summary: str = "",
        confidence: float = 0.0,
        created_at: str = None,
        updated_at: str = None,
        metadata: Dict[str, Any] = None,
        s3_path: str = None,
        aspect: str = "",
        icon: str = "",
        desc_second_view: str = "",
        desc_third_view: str = "",
        content_second_view: str = "",
        content_third_view: str = ""
    ):
        """
        Initialize the L1Shade.
        
        Args:
            id: Unique identifier
            user_id: ID of the user this shade belongs to
            name: Name of the shade
            summary: Short summary of the shade
            confidence: Confidence score (0-1)
            created_at: Creation timestamp
            updated_at: Last update timestamp
            metadata: Additional metadata
            s3_path: Path to detailed data in storage
            aspect: The aspect/category of the shade
            icon: Icon reference for the shade
            desc_second_view: Description in second person
            desc_third_view: Description in third person
            content_second_view: Content in second person
            content_third_view: Content in third person
        """
        self.id = id or str(uuid.uuid4())
        self.user_id = user_id
        self.name = name
        
        # Keep summary for compatibility and map to content_third_view
        self.summary = summary
        self.content_third_view = content_third_view or summary
        
        self.confidence = confidence
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.metadata = metadata or {}
        self.s3_path = s3_path
        
        # Add LPM Kernel compatible fields
        self.aspect = aspect
        self.icon = icon
        self.desc_second_view = desc_second_view
        self.desc_third_view = desc_third_view
        self.content_second_view = content_second_view
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the shade
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "summary": self.summary,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "s3_path": self.s3_path,
            "aspect": self.aspect,
            "icon": self.icon,
            "desc_second_view": self.desc_second_view,
            "desc_third_view": self.desc_third_view,
            "content_second_view": self.content_second_view,
            "content_third_view": self.content_third_view
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L1Shade":
        """
        Create an L1Shade from a dictionary.
        
        Args:
            data: Dictionary representation of a shade
            
        Returns:
            L1Shade object
        """
        return cls(
            id=data.get("id"),
            user_id=data.get("user_id"),
            name=data.get("name", ""),
            summary=data.get("summary", ""),
            confidence=data.get("confidence", 0.0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
            s3_path=data.get("s3_path"),
            aspect=data.get("aspect", ""),
            icon=data.get("icon", ""),
            desc_second_view=data.get("desc_second_view", ""),
            desc_third_view=data.get("desc_third_view", ""),
            content_second_view=data.get("content_second_view", ""),
            content_third_view=data.get("content_third_view", "")
        )
    
    def to_str(self) -> str:
        """
        Format the shade as a string similar to LPM Kernel's ShadeInfo.
        
        Returns:
            String representation of the shade
        """
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Aspect]**: {self.aspect}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.desc_third_view or self.summary}\n\n"
        shade_statement += f"**[Content]**: \n{self.content_third_view or self.summary}\n"
        shade_statement += "---\n\n[Timelines]:\n"
        
        # Add timelines if available in metadata
        if "timelines" in self.metadata:
            for timeline in self.metadata["timelines"]:
                create_time = timeline.get("createTime", "")
                description = timeline.get("description", "")
                ref_id = timeline.get("refId", "")
                shade_statement += f"- {create_time}, {description}, {ref_id}\n"
        
        return shade_statement
        
    def add_second_view(self, domain_desc: str, domain_content: str, domain_timeline: List[Dict[str, Any]]) -> None:
        """
        Add second-person view information to the shade.
        
        Args:
            domain_desc: Second-person description
            domain_content: Second-person content
            domain_timeline: List of timeline entries with second-person descriptions
        """
        self.desc_second_view = domain_desc
        self.content_second_view = domain_content
        
        # Update timelines if available
        if "timelines" in self.metadata and domain_timeline:
            timeline_dict = {
                timeline.get("refId", ""): timeline 
                for timeline in self.metadata["timelines"]
            }
            
            for timeline in domain_timeline:
                ref_id = timeline.get("refId", "")
                if ref_id and ref_id in timeline_dict:
                    # Add second view description to existing timeline
                    timeline_dict[ref_id]["description_second_view"] = timeline.get("description", "")
    
    def improve_shade_info(self, improved_name: str, improved_summary: str, new_timelines: List[Dict[str, Any]]) -> None:
        """
        Improve the shade with new information.
        
        Args:
            improved_name: Updated name if available
            improved_summary: Updated summary/description
            new_timelines: New timeline entries to add
        """
        if improved_name:
            self.name = improved_name
        
        if improved_summary:
            self.summary = improved_summary
            self.content_third_view = improved_summary
            
        # Add new timelines to metadata
        if "timelines" not in self.metadata:
            self.metadata["timelines"] = []
            
        self.metadata["timelines"].extend(new_timelines) 