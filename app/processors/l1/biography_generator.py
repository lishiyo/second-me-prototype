"""
BiographyGenerator for creating user biographies from processed data.

This module provides the BiographyGenerator class that creates user biographies
from processed data, including both global and status biographies, and implements
perspective shifting (first/second/third person).
"""
import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from app.models.l1.bio import Bio
from app.services.llm_service import LLMService
from app.providers.l1.wasabi_adapter import WasabiStorageAdapter

logger = logging.getLogger(__name__)

# LLM Prompt Templates

# This is the equivalent of lpm_kernel's L1/prompt.py's GLOBAL_BIO_SYSTEM_PROMPT
SYS_BIO = """You are a clever and perceptive individual who can, based on a small piece of information from the user, keenly discern some of the user's traits and infer deep insights that are difficult for ordinary people to detect.

The task is to profile the user with the user's interest and characteristics.

Based on the following information sources, construct a comprehensive multi-dimensional profile of the user:
1. Knowledge Shades (representing key aspects of the person's knowledge)
2. Topics and Clusters (representing their areas of interest and focus)

Your analysis should include:
1. A summary of key personality traits
2. An overview of the user's main interests and how they distribute
3. Speculation on the user's likely occupation and other relevant identity information

Please provide both a detailed third-person narrative (500-1000 words) and a concise summary (under 200 words).
"""

# This is our implementation-specific prompt for the user's global biography
USR_BIO = """
I need you to create a comprehensive biography based on the following information sources:

1. Knowledge Shades (representing key aspects of the person's knowledge):
{shades}

2. Topics and Clusters (representing their areas of interest and focus):
{clusters}

Please create a biography with the following components:
1. A comprehensive third-person narrative (500-1000 words) that synthesizes all this information
2. A concise third-person summary (150-200 words) that captures the essence
3. A confidence score (0.0-1.0) reflecting how confident you are in this biography

Format your response as a JSON object with the following structure:
{{
  "content_third_view": "Detailed biography in third person...",
  "summary_third_view": "Concise summary in third person...",
  "confidence": 0.85
}}
"""

SYS_PERSPECTIVE = """You are an expert system for transforming narratives between different grammatical perspectives (first, second, and third person).
Your task is to accurately convert text while maintaining its meaning, tone, and content.
"""

USR_PERSPECTIVE = """
I have a biography written in {source_perspective} perspective that I need to convert to {target_perspective} perspective.

Original {source_perspective} perspective:
{original_text}

Please convert this to {target_perspective} perspective, maintaining the same meaning, details, and tone.
Ensure the transformation is natural and reads well in the new perspective.

Format your response as plain text without any special formatting or additional notes.
"""

# This is the equivalent of lpm_kernel's L1/prompt.py's STATUS_BIO_SYSTEM_PROMPT
SYS_STATUS = """You are intelligent, witty, and possess keen insight. You are very good at analyzing and organizing user's memory.

Based on the person's recent documents, please analyze and create a status biography that does the following:

1. Carefully analyze all the provided documents and construct a three-dimensional and vivid user status report
2. Analyze the specific activities the person has participated in (e.g., attended events, planned activities, expressed interests)
3. Make the report as specific as possible, incorporating entity names or proper nouns from the documents
4. Present each item from a descriptive perspective (e.g., "They did/participated in X" rather than analyzing)
5. Merge similar topics and generate paragraph-style summaries
6. Retain entity names and proper nouns as much as possible
7. Avoid mentioning document types (e.g., "wrote a memo", "recorded audio") - focus on content instead
8. Do not mention specific dates and times in the final content
9. Analyze the person's physical and emotional state changes across their recent activities

Your status biography should focus on recent activities and interests, while creating a coherent narrative that captures the person's current state.
"""

# This is our implementation-specific prompt for the user's status biography
USR_STATUS = """
I need you to create a status biography focusing on the person's recent activities and interests.
This should be based on:

1. Recent Documents:
{documents}

2. Previous Biography (if available):
{previous_bio}

Please create a status biography with the following components:
1. A third-person status narrative (250-500 words) focusing on recent activities
2. A concise third-person summary (100-150 words)
3. A brief analysis of physical and mental health status (under 50 words, from a perspective of care)

Format your response as a JSON object with the following structure:
{{
  "content_third_view": "Status biography in third person...",
  "summary_third_view": "Concise status summary in third person...",
  "health_status": "Brief physical and mental health analysis..."
}}
"""


class BiographyGenerator:
    """
    Creates user biographies from processed data.
    
    This class generates both global and status biographies, implements
    perspective shifting (first/second/third person), and synthesizes
    information from all sources.
    
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
        Initialize the BiographyGenerator.
        
        Args:
            llm_service: Service for LLM interactions
            wasabi_adapter: Adapter for Wasabi storage operations
        """
        self.llm_service = llm_service or LLMService()
        self.wasabi_adapter = wasabi_adapter or WasabiStorageAdapter()
    
    def generate_global_biography(
        self,
        user_id: str,
        old_profile: Bio,
        cluster_list: List[Dict[str, Any]]
    ) -> Bio:
        """
        Generate a global biography.
        
        Args:
            user_id: User ID
            old_profile: Existing biography with shades list
            cluster_list: List of clusters
            
        Returns:
            Generated global biography
        """
        try:
            # Format shades and clusters for the prompt
            shades_text = self._format_shades_for_prompt(old_profile.shades_list)
            clusters_text = self._format_clusters_for_prompt(cluster_list)
            
            # Generate biography using LLM
            messages = [
                {"role": "system", "content": SYS_BIO},
                {"role": "user", "content": USR_BIO.format(
                    shades=shades_text,
                    clusters=clusters_text
                )}
            ]
            
            logger.info(f"Generating global biography for user {user_id}")
            response = self.llm_service.chat_completion(messages)
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response
            bio_data = self._parse_bio_response(content)
            
            if not bio_data or "content_third_view" not in bio_data:
                logger.error(f"Failed to parse biography response: {content}")
                # Use old profile as fallback
                return old_profile
            
            # Create third-person biography
            bio = Bio(
                content_third_view=bio_data.get("content_third_view", ""),
                summary_third_view=bio_data.get("summary_third_view", ""),
                confidence=bio_data.get("confidence", 0.0),
                shades_list=old_profile.shades_list
            )
            
            # Create first and second person views
            bio = self._generate_perspective_shifts(bio)
            
            # Store biography data in Wasabi
            s3_path = self._store_bio_data(user_id, bio.to_dict(), "global")
            
            logger.info(f"Generated global biography with {len(bio.content_third_view)} characters")
            return bio
            
        except Exception as e:
            logger.error(f"Error generating global biography: {str(e)}", exc_info=True)
            # Use old profile as fallback
            return old_profile
    
    def generate_status_biography(
        self,
        user_id: str,
        recent_documents: List[Dict[str, Any]],
        old_bio: Optional[Bio] = None
    ) -> Bio:
        """
        Generate a status biography focusing on recent activity.
        
        Args:
            user_id: User ID
            recent_documents: List of recent documents
            old_bio: Existing biography to update
            
        Returns:
            Generated status biography
        """
        try:
            # Format documents for the prompt
            documents_text = self._format_documents_for_prompt(recent_documents)
            previous_bio_text = old_bio.to_str() if old_bio else ""
            
            # Generate status biography using LLM
            messages = [
                {"role": "system", "content": SYS_STATUS},
                {"role": "user", "content": USR_STATUS.format(
                    documents=documents_text,
                    previous_bio=previous_bio_text
                )}
            ]
            
            logger.info(f"Generating status biography for user {user_id}")
            response = self.llm_service.chat_completion(messages)
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response
            bio_data = self._parse_status_bio_response(content)
            
            if not bio_data or "content_third_view" not in bio_data:
                logger.error(f"Failed to parse status biography response: {content}")
                # Create minimal bio if parsing failed
                bio_data = {
                    "content_third_view": "No recent activity information available.",
                    "summary_third_view": "No recent activity.",
                    "health_status": "No health status analysis available."
                }
            
            # Create third-person biography
            bio = Bio(
                content_third_view=bio_data.get("content_third_view", ""),
                summary_third_view=bio_data.get("summary_third_view", ""),
                health_status=bio_data.get("health_status", "No health status analysis available")
            )
            
            # Create first and second person views
            bio = self._generate_perspective_shifts(bio)
            
            # Store biography data in Wasabi
            s3_path = self._store_bio_data(user_id, bio.to_dict(), "status")
            
            logger.info(f"Generated status biography with {len(bio.content_third_view)} characters")
            return bio
            
        except Exception as e:
            logger.error(f"Error generating status biography: {str(e)}", exc_info=True)
            # Create minimal bio if exception occurred
            return Bio(
                content_third_view="Error generating status biography.",
                summary_third_view="Error generating status.",
                health_status="Error generating health status."
            )
    
    def _generate_perspective_shifts(self, bio: Bio) -> Bio:
        """
        Generate first and second person perspectives from third person biography.
        
        Args:
            bio: Biography object with third-person content
            
        Returns:
            Biography object with all perspectives
        """
        try:
            # Generate first-person content
            first_person_content = self._shift_perspective(
                bio.content_third_view, "third-person", "first-person"
            )
            first_person_summary = self._shift_perspective(
                bio.summary_third_view, "third-person", "first-person"
            )
            
            # Generate second-person content
            second_person_content = self._shift_perspective(
                bio.content_third_view, "third-person", "second-person"
            )
            second_person_summary = self._shift_perspective(
                bio.summary_third_view, "third-person", "second-person"
            )
            
            # Update biography with all perspectives
            bio.content_first_view = first_person_content
            bio.summary_first_view = first_person_summary
            bio.content_second_view = second_person_content
            bio.summary_second_view = second_person_summary
            
            return bio
            
        except Exception as e:
            logger.error(f"Error generating perspective shifts: {str(e)}", exc_info=True)
            # Keep the original third-person views
            return bio
    
    def _shift_perspective(
        self,
        text: str,
        source_perspective: str,
        target_perspective: str
    ) -> str:
        """
        Shift the perspective of a text (first/second/third person).
        
        Args:
            text: Text to shift perspective
            source_perspective: Original perspective
            target_perspective: Target perspective
            
        Returns:
            Text with shifted perspective
        """
        try:
            if not text:
                return ""
                
            messages = [
                {"role": "system", "content": SYS_PERSPECTIVE},
                {"role": "user", "content": USR_PERSPECTIVE.format(
                    source_perspective=source_perspective,
                    target_perspective=target_perspective,
                    original_text=text
                )}
            ]
            
            response = self.llm_service.chat_completion(messages)
            shifted_text = response["choices"][0]["message"]["content"]
            
            return shifted_text
            
        except Exception as e:
            logger.error(f"Error shifting perspective: {str(e)}", exc_info=True)
            return text  # Return original text if shifting fails
    
    def _format_shades_for_prompt(self, shades_list: List[Dict[str, Any]]) -> str:
        """
        Format shades for inclusion in LLM prompt.
        
        Args:
            shades_list: List of shade dictionaries
            
        Returns:
            Formatted shades text
        """
        if not shades_list or len(shades_list) == 0:
            return "No knowledge shades available."
            
        formatted_text = ""
        for i, shade in enumerate(shades_list):
            name = shade.get("name", "Unknown Shade")
            summary = shade.get("summary", "")
            
            formatted_text += f"Shade {i+1}: {name}\n"
            formatted_text += f"Summary: {summary}\n\n"
        
        return formatted_text
    
    def _format_clusters_for_prompt(self, cluster_list: List[Dict[str, Any]]) -> str:
        """
        Format clusters for inclusion in LLM prompt.
        
        Args:
            cluster_list: List of cluster dictionaries
            
        Returns:
            Formatted clusters text
        """
        if not cluster_list or len(cluster_list) == 0:
            return "No topic clusters available."
            
        formatted_text = ""
        for i, cluster in enumerate(cluster_list):
            topic = cluster.get("topic", "Unknown Topic")
            tags = cluster.get("tags", [])
            
            formatted_text += f"Cluster {i+1}: {topic}\n"
            formatted_text += f"Tags: {', '.join(tags)}\n\n"
        
        return formatted_text
    
    def _format_documents_for_prompt(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format documents for inclusion in LLM prompt.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Formatted documents text
        """
        if not documents or len(documents) == 0:
            return "No recent documents available."
            
        formatted_text = ""
        for i, doc in enumerate(documents):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            timestamp = doc.get("created_at", "")
            
            # Limit content length
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            formatted_text += f"Document {i+1}: {title} ({timestamp})\n"
            formatted_text += f"Content: {content}\n\n"
        
        return formatted_text
    
    def _parse_bio_response(self, content: str) -> Dict[str, Any]:
        """
        Parse biography generation response from LLM.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed biography data dictionary
        """
        try:
            # Try to parse as JSON directly
            data = json.loads(content)
            return {
                "content_third_view": data.get("content_third_view", ""),
                "summary_third_view": data.get("summary_third_view", ""),
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
                        "content_third_view": data.get("content_third_view", ""),
                        "summary_third_view": data.get("summary_third_view", ""),
                        "confidence": data.get("confidence", 0.0)
                    }
                else:
                    logger.warning(f"Could not find JSON in response: {content}")
                    return {}
            except Exception as e:
                logger.error(f"Error parsing biography response: {str(e)}", exc_info=True)
                return {}
    
    def _parse_status_bio_response(self, content: str) -> Dict[str, Any]:
        """
        Parse status biography generation response from LLM.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed status biography data dictionary
        """
        try:
            # Try to parse as JSON directly
            data = json.loads(content)
            return {
                "content_third_view": data.get("content_third_view", ""),
                "summary_third_view": data.get("summary_third_view", ""),
                "health_status": data.get("health_status", "No health status analysis available")
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
                        "content_third_view": data.get("content_third_view", ""),
                        "summary_third_view": data.get("summary_third_view", ""),
                        "health_status": data.get("health_status", "No health status analysis available")
                    }
                else:
                    logger.warning(f"Could not find JSON in response: {content}")
                    return {}
            except Exception as e:
                logger.error(f"Error parsing status biography response: {str(e)}", exc_info=True)
                return {}
    
    def _store_bio_data(self, user_id: str, bio_data: Dict[str, Any], bio_type: str) -> str:
        """
        Store biography data in Wasabi.
        
        Args:
            user_id: User ID
            bio_data: Biography data to store
            bio_type: Type of biography ("global" or "status")
            
        Returns:
            S3 path to the stored data
        """
        # Prepare complete biography data
        complete_data = {
            "bio": bio_data,
            "created_at": datetime.now().isoformat()
        }
        
        # Generate a unique path
        bio_id = str(uuid.uuid4())
        s3_path = f"l1/bios/{user_id}/{bio_type}/{bio_id}.json"
        
        # Store in Wasabi
        self.wasabi_adapter.store_json(s3_path, complete_data)
        
        return s3_path 