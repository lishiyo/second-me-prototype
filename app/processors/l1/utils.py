"""
Utility functions for L1 processing.
"""
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, TypeVar, Generic
import json

import numpy as np

# Define a generic type for clusters
T = TypeVar('T')

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_cur_time() -> str:
    """
    Returns the current time formatted as a string.
    
    Returns:
        str: Current time formatted according to TIME_FORMAT.
    """
    cur_time = datetime.now().strftime(TIME_FORMAT)
    return cur_time


def find_connected_components(
    cluster_list: List[T], 
    cluster_merge_distance: float,
    get_center_fn = None
) -> List[List[T]]:
    """
    Finds connected components in a list of clusters based on a distance threshold.
    
    Args:
        cluster_list: List of Cluster objects to analyze
        cluster_merge_distance: Maximum distance for clusters to be considered connected
        get_center_fn: Optional function to get center from cluster object
        
    Returns:
        List[List[Cluster]]: List of connected components, where each component is a list of clusters
    """
    if not cluster_list:
        return []
    
    # Function to get cluster center
    def get_center(cluster, idx):
        if get_center_fn:
            return get_center_fn(cluster)
        elif hasattr(cluster, 'cluster_center'):
            return cluster.cluster_center
        elif hasattr(cluster, 'center_embedding'):
            return cluster.center_embedding
        else:
            # Default behavior if no center attribute is found
            raise ValueError(f"Cluster at index {idx} has no center attribute and no getter function provided")
    
    # Create adjacency matrix
    try:
        adjacency_matrix = np.array([
            [
                np.linalg.norm(
                    np.array(get_center(c1, i)) - 
                    np.array(get_center(c2, j))
                )
                for j, c2 in enumerate(cluster_list)
            ]
            for i, c1 in enumerate(cluster_list)
        ])
    except Exception as e:
        raise ValueError(f"Error creating adjacency matrix: {e}")
    
    # Use BFS to find connected components
    cluster_n = len(cluster_list)
    visited = [False] * cluster_n
    components = []
    
    for i in range(cluster_n):
        if not visited[i]:
            queue = deque([i])
            component = []
            visited[i] = True
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor in range(cluster_n):
                    if (
                        not visited[neighbor] and 
                        adjacency_matrix[node, neighbor] < cluster_merge_distance
                    ):
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components.append([cluster_list[i] for i in component])
    
    return components
    

def is_valid_note(note: Dict[str, Any]) -> bool:
    """
    Checks if a note contains valid creation time information.
    
    Args:
        note: Dictionary containing note data
        
    Returns:
        bool: True if the note has a valid creation time, False otherwise
    """
    if "create_time" in note and note["create_time"]:
        return True
    # Check for lpm_kernel compatibility field
    if "createTime" in note and note["createTime"]:
        return True
    return False


def save_json_without_embeddings(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a JSON file, excluding embedding data to reduce file size.
    
    Args:
        data: Dictionary data to save
        file_path: Path to save the JSON file
    """
    # Create a deep copy to avoid modifying the original data
    data_copy = json.loads(json.dumps(data))
    
    # Recursively remove embedding fields
    def remove_embeddings(obj):
        if isinstance(obj, dict):
            # Remove embedding fields in dictionaries
            if "embedding" in obj:
                del obj["embedding"]
            # Process nested dictionaries
            for key, value in list(obj.items()):
                obj[key] = remove_embeddings(value)
        elif isinstance(obj, list):
            # Process lists
            obj = [remove_embeddings(item) for item in obj]
        return obj
    
    # Clean data
    data_clean = remove_embeddings(data_copy)
    
    # Save to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_clean, f, ensure_ascii=False, indent=2) 