from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np


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
    s3_path: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing for embedding validation"""
        # Validate embedding if present
        if self.embedding is not None:
            if not isinstance(self.embedding, np.ndarray):
                try:
                    # Validate that the embedding is a proper vector
                    if isinstance(self.embedding, dict):
                        raise ValueError(f"Topic {self.id} received embedding in dictionary format, which should have been handled at the boundary")
                        
                    self.embedding = np.array(self.embedding)
                    
                    # Check if it's a scalar (which would be problematic)
                    if self.embedding.shape == ():
                        raise ValueError(f"Topic {self.id} received a scalar embedding. This is invalid for vector operations.")
                except Exception as e:
                    raise ValueError(f"Error converting embedding to numpy array for topic {self.id}: {str(e)}")
                    
            # Make sure embedding is 1D
            if len(self.embedding.shape) > 1:
                try:
                    self.embedding = self.embedding.squeeze()
                    
                    # Final check to ensure we didn't end up with a scalar
                    if self.embedding.shape == ():
                        raise ValueError(f"Squeezing reduced embedding for topic {self.id} to a scalar. Invalid embedding shape.")
                except Exception as e:
                    raise ValueError(f"Error squeezing embedding for topic {self.id}: {str(e)}")
    
    def squeeze(self):
        """
        Return a properly squeezed numpy array of the embedding.
        This ensures vector operations work correctly.
        
        Raises:
            ValueError: If the embedding is a scalar, has invalid shape, or is a dictionary
        """
        if self.embedding is None:
            return None
        
        # Handle dictionary embeddings (unexpected format)
        if isinstance(self.embedding, dict):
            raise ValueError(f"Topic {self.id} has embedding in dictionary format. Cannot use for vector operations.")
            
        # Convert to numpy array
        try:
            embedding_array = np.array(self.embedding)
        except Exception as e:
            raise ValueError(f"Failed to convert embedding to numpy array for topic {self.id}: {str(e)}")
        
        # Check if it's already a proper vector
        if len(embedding_array.shape) >= 1 and embedding_array.shape[0] > 0:
            # Already a proper vector, apply squeeze to remove unnecessary dimensions
            result = embedding_array.squeeze()
            
            # Check if squeeze reduced it to a scalar
            if result.shape == ():
                raise ValueError(f"Squeezing reduced embedding for topic {self.id} to a scalar. Invalid embedding shape.")
                
            return result
            
        # If it's a scalar or problematic shape
        if embedding_array.shape == () or embedding_array.shape[0] == 0:
            raise ValueError(f"Embedding for topic {self.id} has invalid shape {embedding_array.shape}. Cannot use for vector operations.")
            
        return embedding_array
    
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
            "metadata": self.metadata,
            "s3_path": self.s3_path
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
            metadata=data.get("metadata", {}),
            s3_path=data.get("s3_path")
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
    
    def __post_init__(self):
        """Post-initialization processing for embedding validation"""
        # Convert embedding to numpy array if needed
        if self.embedding is not None:
            if not isinstance(self.embedding, np.ndarray):
                try:
                    # Validate that the embedding is a proper vector
                    if isinstance(self.embedding, dict):
                        raise ValueError(f"Memory {self.memory_id} received embedding in dictionary format, which should have been handled at the boundary")
                        
                    self.embedding = np.array(self.embedding)
                    
                    # Check if it's a scalar (which would be problematic)
                    if self.embedding.shape == ():
                        raise ValueError(f"Memory {self.memory_id} received a scalar embedding. This is invalid for vector operations.")
                        
                except Exception as e:
                    raise ValueError(f"Error processing embedding for memory {self.memory_id}: {str(e)}")
                    
            # Make sure embedding is 1D
            if len(self.embedding.shape) > 1:
                try:
                    self.embedding = self.embedding.squeeze()
                    
                    # Final check to ensure we didn't end up with a scalar
                    if self.embedding.shape == ():
                        raise ValueError(f"Squeezing reduced embedding for memory {self.memory_id} to a scalar. Invalid embedding shape.")
                except Exception as e:
                    raise ValueError(f"Error squeezing embedding for memory {self.memory_id}: {str(e)}")
    
    def squeeze(self):
        """
        Return a properly squeezed numpy array of the embedding.
        This ensures vector operations work correctly.
        
        Raises:
            ValueError: If the embedding is a scalar, has invalid shape, or is a dictionary
        """
        if self.embedding is None:
            return None
        
        # Handle dictionary embeddings (unexpected format)
        if isinstance(self.embedding, dict):
            raise ValueError(f"Memory {self.memory_id} has embedding in dictionary format. Cannot use for vector operations.")
            
        # Convert to numpy array
        try:
            embedding_array = np.array(self.embedding)
        except Exception as e:
            raise ValueError(f"Failed to convert embedding to numpy array for memory {self.memory_id}: {str(e)}")
        
        # Check if it's already a proper vector
        if len(embedding_array.shape) >= 1 and embedding_array.shape[0] > 0:
            # Already a proper vector, apply squeeze to remove unnecessary dimensions
            result = embedding_array.squeeze()
            
            # Check if squeeze reduced it to a scalar
            if result.shape == ():
                raise ValueError(f"Squeezing reduced embedding for memory {self.memory_id} to a scalar. Invalid embedding shape.")
                
            return result
            
        # If it's a scalar or problematic shape
        if embedding_array.shape == () or embedding_array.shape[0] == 0:
            raise ValueError(f"Embedding for memory {self.memory_id} has invalid shape {embedding_array.shape}. Cannot use for vector operations.")
            
        return embedding_array
    
    # Add compatibility property for lpm_kernel compatibility
    @property
    def memoryId(self) -> str:
        """Compatibility with lpm_kernel: alias for memory_id"""
        return self.memory_id
    
    @memoryId.setter
    def memoryId(self, value: str):
        """Compatibility with lpm_kernel: alias for memory_id"""
        self.memory_id = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "memoryId": self.memory_id,  # Use memoryId for compatibility
            "embedding": self.embedding,
            "metadata": self.metadata
        }
    
    # Add to_json for compatibility with lpm_kernel
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation for lpm_kernel compatibility"""
        result = {"memoryId": self.memory_id, "embedding": self.embedding}
        # Add all metadata keys to the result
        for k, v in self.metadata.items():
            result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create a Memory from a dictionary"""
        # Check for both our format and lpm_kernel format
        memory_id = data.get("memory_id", data.get("memoryId", ""))
        embedding = data.get("embedding", [])
        
        # Handle potential dict format from Weaviate
        if isinstance(embedding, dict) and 'default' in embedding:
            embedding = embedding['default']
            
        metadata = {k: v for k, v in data.items() 
                  if k not in ["memory_id", "memoryId", "embedding"]}
        
        return cls(
            memory_id=memory_id,
            embedding=embedding,
            metadata=metadata
        )


# Default embedding dimension for compatibility with lpm_kernel
DEFAULT_EMBEDDING_DIM = 1536
# Distance rate for pruning outliers (80% retained)
DISTANCE_RATE = 0.8

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
    s3_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        # Convert any dictionary in memory_list to Memory objects
        memory_list = []
        for memory in self.memory_list:
            if isinstance(memory, dict):
                memory_list.append(Memory.from_dict(memory))
            else:
                memory_list.append(memory)
        self.memory_list = memory_list
        
        # Set merge_list if not present in metadata
        if "merge_list" not in self.metadata:
            self.metadata["merge_list"] = []
            
        # Initialize center_embedding if needed
        if self.center_embedding is None:
            self.get_cluster_center()
        else:
            # Validate center_embedding
            if isinstance(self.center_embedding, dict):
                raise ValueError(f"Cluster {self.id} received center_embedding in dictionary format, which should have been handled at the boundary")
                
            # Ensure it's a proper vector
            try:
                center_embedding_array = np.array(self.center_embedding)
                
                # Check if it's a scalar (which would be problematic)
                if center_embedding_array.shape == ():
                    raise ValueError(f"Cluster {self.id} received a scalar center_embedding. This is invalid for vector operations.")
                    
                # Make sure it's 1D
                if len(center_embedding_array.shape) > 1:
                    center_embedding_array = center_embedding_array.squeeze()
                    
                    # Final check to ensure we didn't end up with a scalar
                    if center_embedding_array.shape == ():
                        raise ValueError(f"Squeezing reduced center_embedding for cluster {self.id} to a scalar. Invalid embedding shape.")
                        
                # Update the center_embedding with the processed version
                self.center_embedding = center_embedding_array.tolist()
            except Exception as e:
                if not isinstance(e, ValueError):  # Don't wrap ValueError again
                    raise ValueError(f"Error processing center_embedding for cluster {self.id}: {str(e)}")
    
    # Add compatibility properties for lpm_kernel compatibility
    @property
    def cluster_id(self) -> str:
        """Compatibility with lpm_kernel: alias for id"""
        return self.id
    
    @cluster_id.setter
    def cluster_id(self, value: str):
        """Compatibility with lpm_kernel: alias for id"""
        self.id = value
    
    @property
    def topic(self) -> str:
        """Compatibility with lpm_kernel: alias for name"""
        return self.name
    
    @topic.setter
    def topic(self, value: str):
        """Compatibility with lpm_kernel: alias for name"""
        self.name = value
    
    @property
    def tags(self) -> List[str]:
        """Compatibility with lpm_kernel: get tags from metadata"""
        return self.metadata.get("tags", [])
    
    @tags.setter
    def tags(self, value: List[str]):
        """Compatibility with lpm_kernel: set tags in metadata"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["tags"] = value
    
    @property
    def is_new(self) -> bool:
        """Compatibility with lpm_kernel: check if cluster is new"""
        return self.metadata.get("is_new", False)
    
    @is_new.setter
    def is_new(self, value: bool):
        """Compatibility with lpm_kernel: set is_new in metadata"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["is_new"] = value
    
    @property
    def cluster_center(self) -> np.ndarray:
        """Get the center embedding of the cluster as numpy array"""
        if self.center_embedding is None:
            # Use zeros as default (similar to lpm_kernel)
            return np.zeros(DEFAULT_EMBEDDING_DIM)
        return np.array(self.center_embedding)
    
    @cluster_center.setter
    def cluster_center(self, value: Union[List[float], np.ndarray]):
        """Set the center embedding of the cluster"""
        if isinstance(value, np.ndarray):
            self.center_embedding = value.tolist()
        else:
            self.center_embedding = value
    
    @property
    def size(self) -> int:
        """Get the size of the cluster (number of memories)"""
        return len(self.memory_list)
    
    @property
    def merge_list(self) -> List[str]:
        """Get list of IDs of merged clusters"""
        return self.metadata.get("merge_list", [])
    
    @merge_list.setter
    def merge_list(self, value: List[str]):
        """Set list of IDs of merged clusters"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["merge_list"] = value
    
    @property
    def document_ids(self) -> List[str]:
        """Get list of memory IDs in this cluster"""
        return [memory.memory_id for memory in self.memory_list]
    
    @property
    def document_count(self) -> int:
        """Get count of memories in this cluster"""
        return len(self.memory_list)
    
    def add_memory(self, memory: Memory) -> None:
        """
        Add a memory to the cluster.
        
        Args:
            memory: Memory to add to the cluster
        """
        self.memory_list.append(memory)
        # Update center embedding
        self.get_cluster_center()
    
    def extend_memory_list(self, memory_list: List[Memory]) -> None:
        """
        Add multiple memories to the cluster.
        
        Args:
            memory_list: List of memories to add to the cluster
        """
        self.memory_list.extend(memory_list)
        # Update center embedding
        self.get_cluster_center()
    
    def get_cluster_center(self) -> np.ndarray:
        """
        Calculate and return the center embedding of the cluster.
        Similar to lpm_kernel's get_cluster_center method.
        
        Returns:
            The center embedding as a numpy array
        """
        if not self.memory_list:
            self.center_embedding = np.zeros(DEFAULT_EMBEDDING_DIM).tolist()
        else:
            memory_embeddings = [np.array(memory.embedding) for memory in self.memory_list
                              if memory.embedding is not None]
            if memory_embeddings:
                self.center_embedding = np.mean(memory_embeddings, axis=0).tolist()
            else:
                self.center_embedding = np.zeros(DEFAULT_EMBEDDING_DIM).tolist()
        
        return np.array(self.center_embedding)
    
    def prune_outliers_from_cluster(self) -> None:
        """
        Remove outlier memories from the cluster.
        Uses lpm_kernel approach: sort by distance from center and keep top DISTANCE_RATE percentage.
        """
        if not self.memory_list:
            self.get_cluster_center()
            return
            
        # Sort memories by distance from center
        center = self.cluster_center
        memory_list = sorted(
            self.memory_list,
            key=lambda x: np.linalg.norm(np.array(x.embedding) - center) if x.embedding is not None else float('inf')
        )
        
        # Keep only the closest DISTANCE_RATE portion
        keep_count = max(int(len(memory_list) * DISTANCE_RATE), 1)
        self.memory_list = memory_list[:keep_count]
        
        # Update center embedding
        self.get_cluster_center()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "clusterId": self.id,  # Use clusterId for compatibility
            "topic": self.name,    # Use topic for compatibility
            "tags": self.tags,     # Include tags for compatibility
            "topic_id": self.topic_id,
            "summary": self.summary,
            "memoryList": [memory.to_json() for memory in self.memory_list],  # Use memoryList for compatibility
            "centerEmbedding": self.center_embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "s3_path": self.s3_path,
            "mergeList": self.merge_list  # Include merge_list for compatibility
        }
    
    # Add to_json for compatibility with lpm_kernel
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation for lpm_kernel compatibility"""
        return {
            "clusterId": self.id if not self.is_new else None,
            # "topic": self.name,
            # "tags": self.tags,
            "memoryList": [memory.to_json() for memory in self.memory_list],
            "centerEmbedding": self.center_embedding,
            "mergeList": self.merge_list
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
        
        # Handle both our format and lpm_kernel format
        cluster_id = data.get("id", data.get("clusterId", ""))
        name = data.get("name", data.get("topic", ""))
        memory_list_data = data.get("memory_list", data.get("memoryList", []))
        center_embedding = data.get("center_embedding", data.get("centerEmbedding"))
        
        metadata = dict(data.get("metadata", {}))
        # Add tags to metadata if present
        if "tags" in data and "tags" not in metadata:
            metadata["tags"] = data.get("tags", [])
        
        # Add merge_list to metadata if present
        if "mergeList" in data and "merge_list" not in metadata:
            metadata["merge_list"] = data.get("mergeList", [])
        
        # Set is_new based on merge_list or explicitly provided value
        if "is_new" not in metadata:
            metadata["is_new"] = data.get("is_new", False)
        
        result = cls(
            id=cluster_id,
            topic_id=data.get("topic_id"),
            name=name,
            summary=data.get("summary"),
            memory_list=[],  # Initialize empty, will add below
            center_embedding=center_embedding,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=metadata,
            s3_path=data.get("s3_path")
        )
        
        # Add memories separately to ensure correct conversion
        for memory_data in memory_list_data:
            if isinstance(memory_data, Memory):
                result.memory_list.append(memory_data)
            else:
                result.memory_list.append(Memory.from_dict(memory_data))
            
        return result

    def squeeze(self):
        """
        Return a properly squeezed numpy array of the center embedding.
        This ensures vector operations work correctly.
        
        Raises:
            ValueError: If the center_embedding is a scalar, has invalid shape, or is a dictionary
        """
        if self.center_embedding is None:
            self.get_cluster_center()
            
        # Handle dictionary embeddings (unexpected format)
        if isinstance(self.center_embedding, dict):
            raise ValueError(f"Cluster {self.id} has center_embedding in dictionary format. Cannot use for vector operations.")
            
        # Convert to numpy array
        try:
            embedding_array = np.array(self.center_embedding)
        except Exception as e:
            raise ValueError(f"Failed to convert center_embedding to numpy array for cluster {self.id}: {str(e)}")
        
        # Check if it's already a proper vector
        if len(embedding_array.shape) >= 1 and embedding_array.shape[0] > 0:
            # Already a proper vector, apply squeeze to remove unnecessary dimensions
            result = embedding_array.squeeze()
            
            # Check if squeeze reduced it to a scalar
            if result.shape == ():
                raise ValueError(f"Squeezing reduced center_embedding for cluster {self.id} to a scalar. Invalid embedding shape.")
                
            return result
            
        # If it's a scalar or problematic shape
        if embedding_array.shape == () or embedding_array.shape[0] == 0:
            raise ValueError(f"Center_embedding for cluster {self.id} has invalid shape {embedding_array.shape}. Cannot use for vector operations.")
            
        return embedding_array 