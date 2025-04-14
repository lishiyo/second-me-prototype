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
SYS_SHADE = """You are a wise, clever person with expertise in data analysis and psychology. You excel at analyzing text and behavioral data, gaining insights into the personal character, qualities, and hobbies of the authors of these texts. Additionally, you possess strong interpersonal skills, allowing you to communicate your insights clearly and effectively.
"""

USR_SHADE = """
I'll provide you with parts of personal private memories, which may include personal creations or online excerpts that reflect certain interests or preferences.

These memories should contain a main component concerning the user's interests or hobbies, ultimately reflecting a certain interest or preference area.

Documents:
{documents}

Your task is to analyze these memories to determine the user's interest or hobby and generate the following content based on that interest:

1. **Name**: A concise name for this knowledge domain or interest (3-6 words)
2. **Summary**: A detailed summary that synthesizes the key information (2-3 paragraphs)
3. **Confidence**: A confidence score (0.0-1.0) reflecting how confident you are in this synthesis
4. **Timelines**: Evolution timeline of the user's interest in this field (where applicable)

For timelines, include:
- createTime: When the event occurred (YYYY-MM-DD format if available)
- description: A brief description of the event
- refId: Reference ID to the original memory or document

Format your response as a JSON object with the following structure:
{{
  "name": "Knowledge Domain Name",
  "summary": "Detailed summary that synthesizes the information...",
  "confidence": 0.85,
  "timelines": [
    {{
      "createTime": "2023-05-15",
      "description": "Brief description of event or milestone",
      "refId": "doc1"
    }}
  ]
}}
"""

SYS_MERGE = """You are a wise, clever person with expertise in data analysis and psychology. You excel at analyzing text and behavioral data, gaining insights into the personal character, qualities, and hobbies of the authors of these texts. Additionally, you possess strong interpersonal skills, allowing you to communicate your insights clearly and effectively.
"""

USR_MERGE = """
I will provide you with multiple analysis contents regarding different areas of a user's interest. These areas are quite similar or have the potential to be merged.

Your task is to:
1. Identify the commonalities among these user interest analyses
2. Extract a more general common interest domain
3. Create a new merged analysis that combines the insights from the original analyses

Shades:
{shades}

Please create a list of merged shades, where each merged shade:
1. Has a concise name (3-6 words)
2. Contains a detailed summary synthesizing the information (2-3 paragraphs)
3. Has a confidence score (0.0-1.0)
4. Includes relevant timelines from the original shades

Format your response as a JSON array of merged shade objects:
[
  {{
    "name": "Merged Interest Domain",
    "summary": "Detailed summary that synthesizes information from multiple shades...",
    "confidence": 0.9,
    "timelines": [
      {{
        "createTime": "2023-05-15",
        "description": "Brief description of event or milestone",
        "refId": "doc1"
      }}
    ]
  }},
  {{
    "name": "Another Merged Domain",
    "summary": "Detailed summary...",
    "confidence": 0.85,
    "timelines": []
  }}
]
"""

SYS_IMPROVE = """You are a wise, clever person with expertise in data analysis and psychology. You excel at analyzing text and behavioral data, gaining insights into the personal character, qualities, and hobbies of the authors of these texts.
"""

USR_IMPROVE = """
I'll provide you with:
1. An existing shade analysis for a specific area of interest
2. Recent memories that may be relevant to this interest area

Existing Shade Info:
{old_shade}

Recent Memories:
{new_memories}

Your task is to update the previous analysis based on these new memories. Please:
1. Determine if the new memories are relevant to the current interest domain
2. If relevant, update the description and content fields as necessary
3. Add new timeline entries for relevant memories

Format your response as a JSON object with the following structure:
{{
  "improved_name": "Updated name if necessary, otherwise keep original", 
  "improved_summary": "Updated summary incorporating new insights", 
  "improved_confidence": 0.85,
  "new_timelines": [
    {{
      "createTime": "2023-05-15",
      "description": "Brief description of new event",
      "refId": "doc1"
    }}
  ]
}}

If no updates are needed, return the original data with an empty new_timelines array.
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
            
            # Prepare metadata with embeddings and timelines
            metadata = {
                "center_embedding": cluster_embedding,
                "cluster_size": cluster_size
            }
            
            # Add timelines to metadata if available
            if "timelines" in shade_data and shade_data["timelines"]:
                metadata["timelines"] = shade_data["timelines"]
            
            # Create shade object - initialize with common attributes
            shade_kwargs = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "name": shade_data.get("name", "Unknown Shade"),
                "summary": shade_data.get("summary", ""),
                "confidence": shade_data.get("confidence", 0.0),
                "metadata": metadata
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
                
                # Gather all timelines from source shades
                all_timelines = []
                for shade in subset_shades:
                    if hasattr(shade, "metadata") and "timelines" in shade.metadata:
                        all_timelines.extend(shade.metadata["timelines"])
                
                # Add any new timelines from the LLM response
                if "timelines" in shade_data and shade_data["timelines"]:
                    all_timelines.extend(shade_data["timelines"])
                
                # Remove potential duplicates (based on refId)
                unique_timelines = []
                seen_ref_ids = set()
                for timeline in all_timelines:
                    ref_id = timeline.get("refId", None)
                    if ref_id and ref_id in seen_ref_ids:
                        continue
                    if ref_id:
                        seen_ref_ids.add(ref_id)
                    unique_timelines.append(timeline)
                    
                # Sort timelines by createTime if available
                sorted_timelines = sorted(
                    unique_timelines,
                    key=lambda x: x.get("createTime", ""),
                    reverse=True
                )
                
                # Store the embedding and timelines in the shade data
                if center_embedding is not None:
                    shade_data["center_embedding"] = center_embedding
                
                shade_data["timelines"] = sorted_timelines
                
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
                        "center_embedding": center_embedding,
                        "timelines": sorted_timelines
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
            formatted_text += f"Summary: {summary}\n"
            
            # Add timelines if available
            if hasattr(shade, 'metadata') and 'timelines' in shade.metadata and shade.metadata['timelines']:
                formatted_text += "Timelines:\n"
                for timeline in shade.metadata['timelines']:
                    create_time = timeline.get('createTime', '')
                    description = timeline.get('description', '')
                    ref_id = timeline.get('refId', '')
                    formatted_text += f"- {create_time}, {description}, {ref_id}\n"
            
            formatted_text += "\n"
        
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
                "confidence": data.get("confidence", 0.0),
                "timelines": data.get("timelines", [])
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
                        "confidence": data.get("confidence", 0.0),
                        "timelines": data.get("timelines", [])
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
                        "confidence": shade.get("confidence", 0.0),
                        "timelines": shade.get("timelines", [])
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
                            "confidence": shade.get("confidence", 0.0),
                            "timelines": shade.get("timelines", [])
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
                "center_embedding": shade_data.get("center_embedding"),
                "timelines": shade_data.get("timelines", [])
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
                "center_embedding": shade_data.get("center_embedding"),
                "timelines": shade_data.get("timelines", [])
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
    
    def improve_shade(
        self, 
        user_id: str,
        old_shade: L1Shade, 
        new_notes: List[Note]
    ) -> L1Shade:
        """
        Improve an existing shade with new notes.
        
        Args:
            user_id: User ID
            old_shade: Existing shade to improve
            new_notes: New notes to incorporate
            
        Returns:
            Improved shade
        """
        try:
            if not new_notes or len(new_notes) == 0:
                logger.warning("No new notes provided for shade improvement")
                return old_shade
            
            # Format the existing shade info and new notes for prompt
            old_shade_text = self._format_shade_for_improvement(old_shade)
            new_notes_text = self._format_notes_for_prompt(new_notes)
            
            # Generate improved shade using LLM
            messages = [
                {"role": "system", "content": SYS_IMPROVE},
                {"role": "user", "content": USR_IMPROVE.format(
                    old_shade=old_shade_text,
                    new_memories=new_notes_text
                )}
            ]
            
            logger.info(f"Improving shade with {len(new_notes)} new notes")
            response = self.llm_service.chat_completion(messages)
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response
            improved_data = self._parse_improved_shade_response(content)
            
            if not improved_data:
                logger.error(f"Failed to parse improved shade response: {content}")
                return old_shade
            
            # Update the shade with improved data
            updated_shade_data = {
                "id": old_shade.id,
                "user_id": user_id,
                "name": improved_data.get("improved_name", old_shade.name),
                "summary": improved_data.get("improved_summary", old_shade.summary),
                "confidence": improved_data.get("improved_confidence", old_shade.confidence),
                "metadata": old_shade.metadata.copy()
            }
            
            # Add s3_path if it exists in the original shade
            if hasattr(old_shade, "s3_path") and old_shade.s3_path:
                updated_shade_data["s3_path"] = old_shade.s3_path
            
            # Add new timelines to metadata if available
            if "timelines" in old_shade.metadata:
                existing_timelines = old_shade.metadata["timelines"]
            else:
                existing_timelines = []
            
            new_timelines = improved_data.get("new_timelines", [])
            updated_shade_data["metadata"]["timelines"] = existing_timelines + new_timelines
            
            # Create new shade object
            improved_shade = L1Shade(**updated_shade_data)
            
            # Store improved shade data if significant changes made
            if improved_data.get("improved_summary") != old_shade.summary or new_timelines:
                improved_shade_dict = {
                    "name": improved_shade.name,
                    "summary": improved_shade.summary,
                    "confidence": improved_shade.confidence,
                    "timelines": updated_shade_data["metadata"].get("timelines", [])
                }
                s3_path = self._store_improved_shade_data(user_id, improved_shade_dict, new_notes)
                
                # Update s3_path if the model accepts it
                if hasattr(improved_shade, "s3_path"):
                    improved_shade.s3_path = s3_path
            
            logger.info(f"Improved shade: {improved_shade.name}")
            return improved_shade
            
        except Exception as e:
            logger.error(f"Error improving shade: {str(e)}", exc_info=True)
            return old_shade
    
    def _format_shade_for_improvement(self, shade: L1Shade) -> str:
        """
        Format a shade for inclusion in improvement prompt.
        
        Args:
            shade: Shade to format
            
        Returns:
            Formatted shade text
        """
        timelines_text = ""
        if hasattr(shade, "metadata") and "timelines" in shade.metadata:
            for i, timeline in enumerate(shade.metadata["timelines"]):
                create_time = timeline.get("createTime", "")
                description = timeline.get("description", "")
                ref_id = timeline.get("refId", "")
                timelines_text += f"- {create_time}, {description}, {ref_id}\n"
        
        formatted_text = f"""Name: {shade.name}
Summary: {shade.summary}
Confidence: {shade.confidence}
Timelines:
{timelines_text}
"""
        return formatted_text
    
    def _parse_improved_shade_response(self, content: str) -> Dict[str, Any]:
        """
        Parse improved shade generation response from LLM.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed improved shade data dictionary
        """
        try:
            # Try to parse as JSON directly
            data = json.loads(content)
            return {
                "improved_name": data.get("improved_name", ""),
                "improved_summary": data.get("improved_summary", ""),
                "improved_confidence": data.get("improved_confidence", 0.0),
                "new_timelines": data.get("new_timelines", [])
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
                        "improved_name": data.get("improved_name", ""),
                        "improved_summary": data.get("improved_summary", ""),
                        "improved_confidence": data.get("improved_confidence", 0.0),
                        "new_timelines": data.get("new_timelines", [])
                    }
                else:
                    logger.warning(f"Could not find JSON in response: {content}")
                    return {}
            except Exception as e:
                logger.error(f"Error parsing improved shade response: {str(e)}", exc_info=True)
                return {}
    
    def _store_improved_shade_data(self, user_id: str, shade_data: Dict[str, Any], notes: List[Note]) -> str:
        """
        Store improved shade data in Wasabi.
        
        Args:
            user_id: User ID
            shade_data: Improved shade data to store
            notes: New notes used for improvement
            
        Returns:
            S3 path to the stored data
        """
        # Prepare complete shade data
        complete_data = {
            "shade": {
                "name": shade_data.get("name", "Unknown Shade"),
                "summary": shade_data.get("summary", ""),
                "confidence": shade_data.get("confidence", 0.0),
                "timelines": shade_data.get("timelines", [])
            },
            "new_notes": [note.to_dict() for note in notes],
            "updated_at": datetime.now().isoformat()
        }
        
        # Generate a unique path
        shade_id = str(uuid.uuid4())
        s3_path = f"l1/improved_shades/{user_id}/{shade_id}.json"
        
        # Store in Wasabi
        self.wasabi_adapter.store_json(s3_path, complete_data)
        
        return s3_path 