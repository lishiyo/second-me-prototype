"""
ShadeGenerator for creating shades from document clusters.

This module provides the ShadeGenerator class that generates "shades" (knowledge aspects)
from document clusters, extracts insights from related documents, and creates coherent
narratives.
"""
import logging
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from app.models.l1.note import Note
from app.models.l1.shade import Shade as L1Shade
from app.services.llm_service import LLMService
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter

logger = logging.getLogger(__name__)

# LLM Prompt Templates
SYS_SHADE = """You are an expert system for extracting coherent knowledge aspects ("shades") from related documents.
Your task is to analyze the provided content and generate a concise summary that captures the key insights and narrative.
"""

USR_SHADE = """
I have a collection of related documents that form a coherent cluster. Please analyze them and generate a knowledge "shade" - 
a coherent narrative that captures the key insights, themes, and knowledge contained in these documents.

Documents:
{documents}

Please create:
1. A concise name for this shade (3-6 words)
2. A detailed summary that synthesizes the key information (2-3 paragraphs)
3. A confidence score (0.0-1.0) reflecting how confident you are in this synthesis

Format your response as a JSON object with the following structure:
{{
  "name": "Shade Name",
  "summary": "Detailed summary that synthesizes the information...",
  "confidence": 0.85
}}
"""

SYS_MERGE = """You are an expert system for merging multiple related knowledge aspects ("shades") into a coherent whole.
Your task is to analyze the provided shades and consolidate them into a unified representation.
"""

USR_MERGE = """
I have multiple knowledge "shades" that need to be merged into a unified representation.
Each shade represents a coherent aspect of knowledge extracted from document clusters.

Shades:
{shades}

Please create:
1. A list of merged shades, where each merged shade:
   a. Has a concise name (3-6 words)
   b. Contains a detailed summary synthesizing the information (2-3 paragraphs)
   c. Has a confidence score (0.0-1.0)

Format your response as a JSON array of shade objects:
[
  {{
    "name": "Merged Shade 1",
    "summary": "Detailed summary...",
    "confidence": 0.9
  }},
  {{
    "name": "Merged Shade 2",
    "summary": "Detailed summary...",
    "confidence": 0.85
  }}
]
"""


class ShadeGenerator:
    """
    Generates "shades" (knowledge aspects) from document clusters.
    
    This class extracts insights from related documents, creates coherent narratives
    from document clusters, and assigns confidence levels to generated insights.
    
    Attributes:
        llm_service: Service for LLM interactions
        wasabi_adapter: Adapter for Wasabi storage operations
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        wasabi_adapter: Optional[WasabiStorageAdapter] = None
    ):
        """
        Initialize the ShadeGenerator.
        
        Args:
            llm_service: Service for LLM interactions
            wasabi_adapter: Adapter for Wasabi storage operations
        """
        self.llm_service = llm_service or LLMService()
        self.wasabi_adapter = wasabi_adapter or WasabiStorageAdapter()
    
    def generate_shade_for_cluster(
        self,
        user_id: str,
        old_shades: List[Dict[str, Any]],
        cluster_notes: List[Note],
        memory_list: List[Dict[str, Any]]
    ) -> Optional[L1Shade]:
        """
        Generate a shade for a cluster.
        
        Args:
            user_id: User ID
            old_shades: List of existing shades
            cluster_notes: List of notes in the cluster
            memory_list: Memory list data
            
        Returns:
            Generated shade or None if generation failed
        """
        try:
            if not cluster_notes or len(cluster_notes) == 0:
                logger.warning("No notes provided for shade generation")
                return None
            
            # Extract cluster embedding data from memory_list if available
            cluster_embedding = None
            cluster_size = len(cluster_notes)
            
            if memory_list and isinstance(memory_list, list):
                for memory in memory_list:
                    if memory.get("cluster_info") and memory.get("notes"):
                        note_ids = [note.id for note in cluster_notes]
                        memory_note_ids = [n.get("id") for n in memory.get("notes", [])]
                        
                        # Check if this memory's notes match our cluster notes
                        if set(note_ids).issubset(set(memory_note_ids)):
                            cluster_embedding = memory.get("cluster_info", {}).get("centerEmbedding")
                            break
            
            # Format documents for prompt
            documents_text = self._format_notes_for_prompt(cluster_notes)
            
            # Generate shade using LLM
            messages = [
                {"role": "system", "content": SYS_SHADE},
                {"role": "user", "content": USR_SHADE.format(documents=documents_text)}
            ]
            
            logger.info(f"Generating shade for cluster with {len(cluster_notes)} notes")
            response = self.llm_service.chat_completion(messages)
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response
            shade_data = self._parse_shade_response(content)
            
            if not shade_data or "name" not in shade_data:
                logger.error(f"Failed to parse shade response: {content}")
                return None
                
            # Add cluster embedding to shade data if available
            if cluster_embedding:
                shade_data["center_embedding"] = cluster_embedding
            
            # Store shade data in Wasabi
            s3_path = self._store_shade_data(user_id, shade_data, cluster_notes)
            
            # Create shade object - initialize with common attributes
            shade_kwargs = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "name": shade_data.get("name", "Unknown Shade"),
                "summary": shade_data.get("summary", ""),
                "confidence": shade_data.get("confidence", 0.0),
                "metadata": {
                    "center_embedding": cluster_embedding,
                    "cluster_size": cluster_size
                }
            }
            
            # Only add s3_path if the class accepts it (inspect the class's __init__ parameters)
            import inspect
            if 's3_path' in inspect.signature(L1Shade.__init__).parameters:
                shade_kwargs["s3_path"] = s3_path
            
            # Create shade object
            shade = L1Shade(**shade_kwargs)
            
            logger.info(f"Generated shade: {shade.name} with confidence {shade.confidence}")
            return shade
            
        except Exception as e:
            logger.error(f"Error generating shade: {str(e)}", exc_info=True)
            return None
    
    def merge_shades(
        self,
        user_id: str,
        shades: List[L1Shade]
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple shades into a coherent representation.
        
        Args:
            user_id: User ID
            shades: List of shades to merge
            
        Returns:
            List of merged shade dictionaries
        """
        try:
            if not shades or len(shades) == 0:
                logger.warning("No shades to merge")
                return []
            
            # Single shade doesn't need merging
            if len(shades) == 1:
                return [shades[0].to_dict() if hasattr(shades[0], 'to_dict') else shades[0]]
            
            # Format shades for prompt
            shades_text = self._format_shades_for_prompt(shades)
            
            # Generate merged shades using LLM
            messages = [
                {"role": "system", "content": SYS_MERGE},
                {"role": "user", "content": USR_MERGE.format(shades=shades_text)}
            ]
            
            logger.info(f"Merging {len(shades)} shades")
            response = self.llm_service.chat_completion(messages)
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response
            merged_shades_data = self._parse_merged_shades_response(content)
            
            if not merged_shades_data or len(merged_shades_data) == 0:
                logger.error(f"Failed to parse merged shades response: {content}")
                return [shade.to_dict() if hasattr(shade, 'to_dict') else shade for shade in shades]
            
            # Store merged shades in Wasabi and create shade objects
            result_shades = []
            
            # Check if s3_path is used in the L1Shade model
            import inspect
            includes_s3_path = 's3_path' in inspect.signature(L1Shade.__init__).parameters
            
            for shade_data in merged_shades_data:
                # Get the subset of shades that should be merged into this new shade
                # In this simplified implementation, we merge all shades into each result
                # In a production environment, we would need a way to determine which 
                # original shades belong to which merged shade
                subset_shades = shades
                
                # Calculate center embedding for the merged shade if embeddings are available
                center_embedding = self._calculate_merged_center_embedding(subset_shades)
                
                # Store the embedding in the shade metadata if available
                if center_embedding is not None:
                    shade_data["center_embedding"] = center_embedding
                
                s3_path = self._store_merged_shade_data(user_id, shade_data)
                
                # Create the merged shade dictionary
                shade_dict = {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "name": shade_data.get("name", "Unknown Shade"),
                    "summary": shade_data.get("summary", ""),
                    "confidence": shade_data.get("confidence", 0.0),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "metadata": {
                        "source_shades": [s.id for s in subset_shades],
                        "center_embedding": center_embedding
                    }
                }
                
                # Only add s3_path if it's used in the model
                if includes_s3_path:
                    shade_dict["s3_path"] = s3_path
                
                result_shades.append(shade_dict)
            
            logger.info(f"Generated {len(result_shades)} merged shades")
            return result_shades
            
        except Exception as e:
            logger.error(f"Error merging shades: {str(e)}", exc_info=True)
            return [shade.to_dict() if hasattr(shade, 'to_dict') else shade for shade in shades]
    
    def _format_notes_for_prompt(self, notes: List[Note]) -> str:
        """
        Format notes for inclusion in LLM prompt.
        
        Args:
            notes: List of notes to format
            
        Returns:
            Formatted notes text
        """
        formatted_text = ""
        for i, note in enumerate(notes):
            formatted_text += f"Document {i+1}: {note.title}\n"
            formatted_text += f"Content: {note.content[:1000]}...\n\n"  # Limit content length
        
        return formatted_text
    
    def _format_shades_for_prompt(self, shades: List[L1Shade]) -> str:
        """
        Format shades for inclusion in LLM prompt.
        
        Args:
            shades: List of shades to format
            
        Returns:
            Formatted shades text
        """
        formatted_text = ""
        for i, shade in enumerate(shades):
            name = shade.name if hasattr(shade, 'name') else "Unknown Shade"
            summary = shade.summary if hasattr(shade, 'summary') else ""
            
            formatted_text += f"Shade {i+1}: {name}\n"
            formatted_text += f"Summary: {summary}\n\n"
        
        return formatted_text
    
    def _parse_shade_response(self, content: str) -> Dict[str, Any]:
        """
        Parse shade generation response from LLM.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed shade data dictionary
        """
        try:
            # Try to parse as JSON directly
            data = json.loads(content)
            return {
                "name": data.get("name", "Unknown Shade"),
                "summary": data.get("summary", ""),
                "confidence": data.get("confidence", 0.0)
            }
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON part
            try:
                # Find JSON-like structure in the text
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                    return {
                        "name": data.get("name", "Unknown Shade"),
                        "summary": data.get("summary", ""),
                        "confidence": data.get("confidence", 0.0)
                    }
                else:
                    logger.warning(f"Could not find JSON in response: {content}")
                    return {}
            except Exception as e:
                logger.error(f"Error parsing shade response: {str(e)}", exc_info=True)
                return {}
    
    def _parse_merged_shades_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse merged shades response from LLM.
        
        Args:
            content: LLM response content
            
        Returns:
            List of parsed shade data dictionaries
        """
        try:
            # Try to parse as JSON directly
            data = json.loads(content)
            if isinstance(data, list):
                return [
                    {
                        "name": shade.get("name", "Unknown Shade"),
                        "summary": shade.get("summary", ""),
                        "confidence": shade.get("confidence", 0.0)
                    }
                    for shade in data
                ]
            return []
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON part
            try:
                # Find JSON-like structure in the text
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                    return [
                        {
                            "name": shade.get("name", "Unknown Shade"),
                            "summary": shade.get("summary", ""),
                            "confidence": shade.get("confidence", 0.0)
                        }
                        for shade in data
                    ]
                else:
                    logger.warning(f"Could not find JSON array in response: {content}")
                    return []
            except Exception as e:
                logger.error(f"Error parsing merged shades response: {str(e)}", exc_info=True)
                return []
    
    def _store_shade_data(self, user_id: str, shade_data: Dict[str, Any], notes: List[Note]) -> str:
        """
        Store shade data in Wasabi.
        
        Args:
            user_id: User ID
            shade_data: Shade data to store
            notes: Notes associated with the shade
            
        Returns:
            S3 path to the stored data
        """
        # Prepare complete shade data including notes
        complete_data = {
            "shade": {
                "name": shade_data.get("name", "Unknown Shade"),
                "summary": shade_data.get("summary", ""),
                "confidence": shade_data.get("confidence", 0.0),
                "center_embedding": shade_data.get("center_embedding")
            },
            "notes": [note.to_dict() for note in notes],
            "created_at": datetime.now().isoformat()
        }
        
        # Generate a unique path
        shade_id = str(uuid.uuid4())
        s3_path = f"l1/shades/{user_id}/{shade_id}.json"
        
        # Store in Wasabi
        self.wasabi_adapter.store_json(s3_path, complete_data)
        
        return s3_path
    
    def _store_merged_shade_data(self, user_id: str, shade_data: Dict[str, Any]) -> str:
        """
        Store merged shade data in Wasabi.
        
        Args:
            user_id: User ID
            shade_data: Merged shade data to store
            
        Returns:
            S3 path to the stored data
        """
        # Prepare complete shade data
        complete_data = {
            "shade": {
                "name": shade_data.get("name", "Unknown Shade"),
                "summary": shade_data.get("summary", ""),
                "confidence": shade_data.get("confidence", 0.0),
                "center_embedding": shade_data.get("center_embedding")
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Generate a unique path
        shade_id = str(uuid.uuid4())
        s3_path = f"l1/merged_shades/{user_id}/{shade_id}.json"
        
        # Store in Wasabi
        self.wasabi_adapter.store_json(s3_path, complete_data)
        
        return s3_path
    
    def _calculate_merged_center_embedding(self, shades: List[L1Shade]) -> Optional[List[float]]:
        """
        Calculate the center embedding for merged shades.
        
        This method calculates a weighted average of center embeddings from 
        the input shades based on cluster size.
        
        Args:
            shades: List of shades to merge
            
        Returns:
            List of floats representing the center embedding or None if embeddings unavailable
        """
        try:
            # Check if embeddings exist in shade metadata
            embeddings_exist = all('center_embedding' in (shade.metadata if hasattr(shade, 'metadata') else {}) 
                                for shade in shades)
            
            if not embeddings_exist:
                logger.warning("Embeddings not found in one or more shades")
                return None
                
            # Get the first shade's embedding to determine vector dimensions
            first_embedding = shades[0].metadata.get('center_embedding')
            if not first_embedding or not isinstance(first_embedding, list):
                logger.warning(f"Invalid embedding format: {first_embedding}")
                return None
                
            # Initialize total embedding and cluster size
            total_embedding = np.zeros(len(first_embedding))
            total_cluster_size = 0
            
            # Sum weighted embeddings
            for shade in shades:
                # Get cluster size (default to 1 if not specified)
                cluster_size = shade.metadata.get('cluster_size', 1)
                
                # Get center embedding
                center_embedding = shade.metadata.get('center_embedding')
                if not center_embedding or not isinstance(center_embedding, list):
                    continue
                    
                # Add weighted embedding to total
                try:
                    embedding_array = np.array(center_embedding)
                    total_embedding += cluster_size * embedding_array
                    total_cluster_size += cluster_size
                except Exception as e:
                    logger.error(f"Error processing embedding: {str(e)}")
                    continue
            
            # Return the average embedding
            if total_cluster_size > 0:
                return (total_embedding / total_cluster_size).tolist()
            else:
                logger.warning("Total cluster size is zero")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating merged center embedding: {str(e)}", exc_info=True)
            return None 