from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json


@dataclass
class ShadeTimeline:
    """
    Represents a timeline entry for a shade, matching lpm_kernel's implementation.
    """
    refMemoryId: str = None  # Can be either refMemoryId or refId
    createTime: str = ""
    descSecondView: str = ""
    descThirdView: str = ""
    isNew: bool = True
    
    # Add properties for snake_case compatibility
    @property
    def ref_memory_id(self) -> str:
        """Get reference memory ID."""
        return self.refMemoryId
        
    @property
    def create_time(self) -> str:
        """Get creation time."""
        return self.createTime
        
    @property
    def desc_second_view(self) -> str:
        """Get second-person description."""
        return self.descSecondView
        
    @property
    def desc_third_view(self) -> str:
        """Get third-person description."""
        return self.descThirdView
        
    @property
    def is_new(self) -> bool:
        """Check if timeline is new."""
        return self.isNew
    
    @classmethod
    def from_raw_format(cls, raw_format: Dict[str, Any]) -> "ShadeTimeline":
        """
        Create a ShadeTimeline from a raw format dictionary.
        Handles both our field naming and lpm_kernel's field naming.
        
        Args:
            raw_format: Dictionary with timeline data
            
        Returns:
            ShadeTimeline object
        """
        # Handle different field naming conventions
        ref_id = raw_format.get("refMemoryId", raw_format.get("refId", ""))
        desc_third_view = raw_format.get("descThirdView", raw_format.get("description", ""))
        desc_second_view = raw_format.get("descSecondView", raw_format.get("description_second_view", ""))
        
        return cls(
            refMemoryId=ref_id,
            createTime=raw_format.get("createTime", ""),
            descSecondView=desc_second_view,
            descThirdView=desc_third_view,
            isNew=raw_format.get("isNew", True)
        )
    
    def add_second_view(self, description: str) -> None:
        """
        Add second-person view description.
        
        Args:
            description: Second-person description
        """
        self.descSecondView = description
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation using our field naming.
        
        Returns:
            Dictionary representation of the timeline
        """
        return {
            "createTime": self.createTime,
            "refMemoryId": self.refMemoryId,
            "refId": self.refMemoryId,  # Include both for compatibility
            "description": self.descThirdView,
            "description_second_view": self.descSecondView,
            "isNew": self.isNew
        }
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON representation using lpm_kernel's field naming.
        
        Returns:
            Dictionary representation matching lpm_kernel's format
        """
        return {
            "createTime": self.createTime,
            "refMemoryId": self.refMemoryId,
            "descThirdView": self.descThirdView,
            "descSecondView": self.descSecondView
        }


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
    descThirdView: str = ""
    contentThirdView: str = ""
    descSecondView: str = ""
    contentSecondView: str = ""
    
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
            "descThirdView": self.descThirdView,
            "contentThirdView": self.contentThirdView,
            "descSecondView": self.descSecondView,
            "contentSecondView": self.contentSecondView
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
            descThirdView=data.get("descThirdView", data.get("desc_third_view", "")),
            contentThirdView=data.get("contentThirdView", data.get("content_third_view", "")),
            descSecondView=data.get("descSecondView", data.get("desc_second_view", "")),
            contentSecondView=data.get("contentSecondView", data.get("content_second_view", ""))
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
            descThirdView=shade.desc_third_view,
            contentThirdView=shade.content_third_view,
            descSecondView=shade.desc_second_view,
            contentSecondView=shade.content_second_view
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
            desc_third_view=self.descThirdView,
            content_third_view=self.contentThirdView or self.content,
            desc_second_view=self.descSecondView,
            content_second_view=self.contentSecondView
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
    descThirdView: str = ""
    contentThirdView: str = ""
    descSecondView: str = ""
    contentSecondView: str = ""
    cluster_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timelines: List[ShadeTimeline] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation using our field naming"""
        return {
            "shade_id": self.shade_id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "descThirdView": self.descThirdView,
            "contentThirdView": self.contentThirdView,
            "descSecondView": self.descSecondView,
            "contentSecondView": self.contentSecondView,
            "cluster_info": self.cluster_info,
            "metadata": self.metadata,
            "timelines": [t.to_dict() for t in self.timelines]
        }
        
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation using lpm_kernel's field naming"""
        result = {
            "id": self.shade_id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "descThirdView": self.descThirdView,
            "contentThirdView": self.contentThirdView,
            "descSecondView": self.descSecondView,
            "contentSecondView": self.contentSecondView,
        }
        if self.timelines:
            result["timelines"] = [t.to_json() for t in self.timelines]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadeMergeInfo":
        """Create a ShadeMergeInfo from a dictionary"""
        result = cls(
            shade_id=data.get("shade_id", ""),
            name=data.get("name", ""),
            aspect=data.get("aspect", ""),
            icon=data.get("icon", ""),
            descThirdView=data.get("descThirdView", data.get("desc_third_view", "")),
            contentThirdView=data.get("contentThirdView", data.get("content_third_view", "")),
            descSecondView=data.get("descSecondView", data.get("desc_second_view", "")),
            contentSecondView=data.get("contentSecondView", data.get("content_second_view", "")),
            cluster_info=data.get("cluster_info"),
            metadata=data.get("metadata", {}),
            timelines=[]
        )
        
        # Process timelines if present
        if "timelines" in data:
            result.timelines = [ShadeTimeline.from_raw_format(t) for t in data["timelines"]]
        
        return result
    
    @classmethod
    def from_shade(cls, shade: "L1Shade") -> "ShadeMergeInfo":
        """Create a ShadeMergeInfo from an L1Shade"""
        return cls(
            shade_id=shade.id,
            name=shade.name,
            aspect=shade.aspect,
            icon=shade.icon,
            descThirdView=shade.desc_third_view,
            contentThirdView=shade.content_third_view,
            descSecondView=shade.desc_second_view,
            contentSecondView=shade.content_second_view,
            metadata=shade.metadata,
            timelines=shade.timelines.copy()
        )
        
    def improve_shade_info(self, improved_desc: str, improved_content: str) -> None:
        """Improve the shade with new description and content"""
        self.descThirdView = improved_desc
        self.contentThirdView = improved_content
        
    def add_second_view(self, domain_desc: str, domain_content: str) -> None:
        """Add second-person view information"""
        self.descSecondView = domain_desc
        self.contentSecondView = domain_content
        
    def to_str(self) -> str:
        """Format the shade as a string"""
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Aspect]**: {self.aspect}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.descThirdView}\n\n**[Content]**: \n{self.contentThirdView}\n"
        shade_statement += "---\n\n"
        
        if self.timelines:
            shade_statement += "[Timelines]:\n"
            for timeline in self.timelines:
                shade_statement += f"- {timeline.create_time}, {timeline.descThirdView}, {timeline.ref_memory_id}\n"
        
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
        descSecondView: Description in second person
        descThirdView: Description in third person
        contentSecondView: Content in second person
        contentThirdView: Content in third person
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
        descSecondView: str = "",
        descThirdView: str = "",
        contentSecondView: str = "",
        contentThirdView: str = "",
        # Support snake_case for backward compatibility
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
            descSecondView: Description in second person (camelCase)
            descThirdView: Description in third person (camelCase)
            contentSecondView: Content in second person (camelCase)
            contentThirdView: Content in third person (camelCase)
            desc_second_view: Description in second person (snake_case, for compatibility)
            desc_third_view: Description in third person (snake_case, for compatibility)
            content_second_view: Content in second person (snake_case, for compatibility)
            content_third_view: Content in third person (snake_case, for compatibility)
        """
        self.id = id or str(uuid.uuid4())
        self.user_id = user_id
        self.name = name
        
        # Keep summary for compatibility and map to content_third_view
        self.summary = summary
        
        # Handle both camelCase and snake_case inputs
        self.desc_second_view = descSecondView or desc_second_view
        self.desc_third_view = descThirdView or desc_third_view
        self.content_second_view = contentSecondView or content_second_view
        self.content_third_view = contentThirdView or content_third_view or summary
        
        self.confidence = confidence
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.metadata = metadata or {}
        self.s3_path = s3_path
        
        # Add LPM Kernel compatible fields
        self.aspect = aspect
        self.icon = icon
        
        # Add timelines property that uses ShadeTimeline objects
        self._timelines = []
        if "timelines" in self.metadata:
            # Handle both our field naming and lpm_kernel field naming
            self._timelines = []
            for t in self.metadata["timelines"]:
                # Check if this is our format (has "description") or lpm_kernel's (has "descThirdView")
                if "descThirdView" in t:
                    # Convert from lpm_kernel format
                    t = {
                        "createTime": t.get("createTime", ""),
                        "refMemoryId": t.get("refMemoryId"),
                        "refId": t.get("refMemoryId"),  # Use refMemoryId as refId
                        "description": t.get("descThirdView", ""),
                        "description_second_view": t.get("descSecondView", ""),
                        "isNew": True
                    }
                self._timelines.append(ShadeTimeline.from_raw_format(t))
    
    @property
    def timelines(self) -> List[ShadeTimeline]:
        """
        Get shade timelines.
        
        Returns:
            List of ShadeTimeline objects
        """
        return self._timelines
    
    # Add camelCase property accessors for LPM Kernel compatibility
    @property
    def descSecondView(self) -> str:
        """Get second-person description in camelCase style."""
        return self.desc_second_view
    
    @property
    def descThirdView(self) -> str:
        """Get third-person description in camelCase style."""
        return self.desc_third_view
    
    @property
    def contentSecondView(self) -> str:
        """Get second-person content in camelCase style."""
        return self.content_second_view
    
    @property
    def contentThirdView(self) -> str:
        """Get third-person content in camelCase style."""
        return self.content_third_view
    
    def _sync_timelines_to_metadata(self) -> None:
        """
        Sync timeline objects to metadata dictionary for storage.
        Uses our field naming convention for storage.
        """
        self.metadata["timelines"] = [t.to_dict() for t in self._timelines]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the shade
        """
        # Ensure timelines are synced to metadata
        self._sync_timelines_to_metadata()
        
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
            "descSecondView": self.desc_second_view,
            "descThirdView": self.desc_third_view,
            "contentSecondView": self.content_second_view,
            "contentThirdView": self.content_third_view
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
            # Handle both camelCase and snake_case field names
            descSecondView=data.get("descSecondView", data.get("desc_second_view", "")),
            descThirdView=data.get("descThirdView", data.get("desc_third_view", "")),
            contentSecondView=data.get("contentSecondView", data.get("content_second_view", "")),
            contentThirdView=data.get("contentThirdView", data.get("content_third_view", ""))
        )
    
    def to_str(self) -> str:
        """
        Format the shade as a string similar to LPM Kernel's ShadeInfo.
        
        Returns:
            String representation of the shade
        """
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Aspect]**: {self.aspect}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.descThirdView or self.summary}\n\n"
        shade_statement += f"**[Content]**: \n{self.contentThirdView or self.summary}\n"
        shade_statement += "---\n\n[Timelines]:\n"
        
        # Add timelines using ShadeTimeline objects
        for timeline in self.timelines:
            shade_statement += f"- {timeline.createTime}, {timeline.descThirdView}, {timeline.refMemoryId}\n"
        
        return shade_statement
        
    def add_second_view(self, domainDesc: str, domainContent: str, domainTimeline: List[Dict[str, Any]]) -> None:
        """
        Add second-person view information to the shade.
        
        Args:
            domainDesc: Second-person description
            domainContent: Second-person content
            domainTimeline: List of timeline entries with second-person descriptions
        """
        self.desc_second_view = domainDesc
        self.content_second_view = domainContent
        
        # Create a lookup dictionary for timeline objects
        timeline_dict = {t.ref_memory_id: t for t in self.timelines}
        
        # Process domain_timeline to update second-person views
        for timeline_data in domainTimeline:
            ref_id = timeline_data.get("refMemoryId", timeline_data.get("refId", ""))
            if ref_id and ref_id in timeline_dict:
                # Add second-person view description to existing timeline
                timeline_dict[ref_id].add_second_view(timeline_data.get("description", ""))
        
        # Sync changes back to metadata
        self._sync_timelines_to_metadata()
    
    def improve_shade_info(self, improvedName: str, improvedDesc: str, improvedTimelines: List[Dict[str, Any]]) -> None:
        """
        Improve the shade with new information.
        
        Args:
            improvedName: Updated name if available
            improvedDesc: Updated summary/description
            improvedTimelines: New timeline entries to add
        """
        if improvedName:
            self.name = improvedName
        
        if improvedDesc:
            self.summary = improvedDesc
            self.desc_third_view = improvedDesc
            self.content_third_view = improvedDesc
            
        # Add new timelines as ShadeTimeline objects
        for timeline_data in improvedTimelines:
            self._timelines.append(ShadeTimeline.from_raw_format(timeline_data))
        
        # Sync changes back to metadata
        self._sync_timelines_to_metadata()
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON format using lpm_kernel's field naming.
        
        Returns:
            Dictionary representation matching lpm_kernel's format
        """
        return {
            "id": self.id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "descThirdView": self.desc_third_view,
            "contentThirdView": self.content_third_view,
            "descSecondView": self.desc_second_view,
            "contentSecondView": self.content_second_view,
            "timelines": [t.to_json() for t in self.timelines]
        } 