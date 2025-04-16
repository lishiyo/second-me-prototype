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
    tags: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing for embedding validation"""
        # Validate embedding and make sure it's a numpy array
        if self.embedding is not None:
            # Convert embedding to numpy array if it's not already
            if not isinstance(self.embedding, np.ndarray):
                try:
                    # Validate that the embedding is a proper vector
                    if isinstance(self.embedding, dict):
                        raise ValueError(f"Chunk {self.id} received embedding in dictionary format, which should have been handled at the boundary")
                        
                    self.embedding = np.array(self.embedding)
                    
                    # Check if it's a scalar (which would be problematic)
                    if self.embedding.shape == ():
                        raise ValueError(f"Chunk {self.id} received a scalar embedding. This is invalid for vector operations.")
                except Exception as e:
                    raise ValueError(f"Error converting embedding to numpy array for chunk {self.id}: {str(e)}")
                    
            # Make sure embedding is 1D
            if len(self.embedding.shape) > 1:
                try:
                    self.embedding = self.embedding.squeeze()
                    
                    # Final check to ensure we didn't end up with a scalar
                    if self.embedding.shape == ():
                        raise ValueError(f"Squeezing reduced embedding for chunk {self.id} to a scalar. Invalid embedding shape.")
                except Exception as e:
                    raise ValueError(f"Error squeezing embedding for chunk {self.id}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "document_id": self.document_id,
            "tags": self.tags,
            "topic": self.topic,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        """
        Create a Chunk from a dictionary.
        
        Args:
            data: Dictionary with chunk data
        
        Returns:
            Chunk object
        """
        chunk_id = data.get("id", None)
        document_id = data.get("document_id", None)
        content = data.get("content", "")
        tags = data.get("tags", [])
        topic = data.get("topic", "")
        embedding = data.get("embedding", None)
        chunk_index = data.get("chunk_index", 0)
        metadata = data.get("metadata", {})
        
        # No longer need to handle dict embeddings here
                
        return cls(
            id=chunk_id,
            document_id=document_id,
            content=content,
            embedding=embedding,
            tags=tags,
            topic=topic,
            chunk_index=chunk_index,
            metadata=metadata
        )
    
    # For compatibility with lpm_kernel Chunk class
    def squeeze(self):
        """
        Properly squeeze embedding while preserving vector structure.
        Returns a numpy array representation of the embedding.
        
        Raises:
            ValueError: If the embedding is a scalar or has invalid shape
        """
        if self.embedding is None:
            return None
        
        # Convert to numpy array
        embedding_array = np.array(self.embedding)
        
        # Check shape to ensure we don't lose dimensions
        if embedding_array.shape == ():  # Scalar value
            # This is an error - raise an exception
            raise ValueError(f"Embedding for chunk {self.id} is a scalar value. Cannot use this for vector operations.")
            
        # If it's already a proper vector with at least one dimension, just return it
        if len(embedding_array.shape) >= 1:
            # Apply squeeze only if there are unnecessary dimensions (e.g., [[1,2,3]] -> [1,2,3])
            result = embedding_array.squeeze()
            
            # Ensure we didn't squeeze it to a scalar
            if result.shape == ():
                raise ValueError(f"Squeezing reduced embedding for chunk {self.id} to a scalar. Invalid embedding shape.")
                
            return result
            
        # Should not reach here, but just in case
        raise ValueError(f"Unexpected embedding shape {embedding_array.shape} for chunk {self.id}")


@dataclass
class Note:
    """
    Represents a processed document with metadata and embeddings.
    Similar to the Note class in lpm_kernel.
    """
    id: str
    content: str
    create_time: Union[datetime, str]
    embedding: Optional[np.ndarray] = None
    chunks: List[Chunk] = field(default_factory=list)
    title: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    insight: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    memory_type: str = "TEXT"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing for embedding validation"""
        if self.embedding is not None:
            # Convert to numpy array if it's not already
            if not isinstance(self.embedding, np.ndarray):
                try:
                    # Validate that the embedding is a proper vector
                    if isinstance(self.embedding, dict):
                        raise ValueError(f"Note {self.id} received embedding in dictionary format, which should have been handled at the boundary")
                        
                    self.embedding = np.array(self.embedding)
                    
                    # Check if it's a scalar (which would be problematic)
                    if self.embedding.shape == ():
                        raise ValueError(f"Note {self.id} received a scalar embedding. This is invalid for vector operations.")
                except Exception as e:
                    raise ValueError(f"Error converting embedding to numpy array for note {self.id}: {str(e)}")
            
            # Make sure embedding is 1D
            if len(self.embedding.shape) > 1:
                try:
                    self.embedding = self.embedding.squeeze()
                    
                    # Final check to ensure we didn't end up with a scalar
                    if self.embedding.shape == ():
                        raise ValueError(f"Squeezing reduced embedding for note {self.id} to a scalar. Invalid embedding shape.")
                except Exception as e:
                    raise ValueError(f"Error squeezing embedding for note {self.id}: {str(e)}")
    
    # Properties to maintain compatibility with lpm_kernel Note class
    @property
    def noteId(self) -> str:
        """Compatibility with lpm_kernel: alias for id"""
        return self.id
        
    @noteId.setter
    def noteId(self, value: str):
        """Compatibility with lpm_kernel: alias for id"""
        self.id = value
        
    @property
    def createTime(self) -> str:
        """Compatibility with lpm_kernel: alias for create_time"""
        if isinstance(self.create_time, datetime):
            return self.create_time.strftime("%Y-%m-%d %H:%M:%S")
        return self.create_time
        
    @createTime.setter
    def createTime(self, value: str):
        """Compatibility with lpm_kernel: alias for create_time"""
        self.create_time = value
        
    @property
    def memoryType(self) -> str:
        """Compatibility with lpm_kernel: alias for memory_type"""
        return self.memory_type
        
    @memoryType.setter
    def memoryType(self, value: str):
        """Compatibility with lpm_kernel: alias for memory_type"""
        self.memory_type = value
    
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
        
    # Add a to_json method for compatibility with lpm_kernel Note class
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the note to a JSON-serializable dictionary.
        Compatibility with lpm_kernel Note class.
        """
        if hasattr(self, "processed"):
            return {
                "id": self.id,
                "insight": self.insight,
                "summary": self.summary,
                "memory_type": self.memory_type,
                "create_time": self.createTime,
                "title": self.title,
                "content": self.content,
                "processed": self.processed
            }
        else:
            return {
                "id": self.id,
                "insight": self.insight,
                "summary": self.summary,
                "memory_type": self.memory_type,
                "create_time": self.createTime,
                "title": self.title,
                "content": self.content
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