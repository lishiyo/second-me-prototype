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
from app.models.l1.shade import L1Shade
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

SYS_MERGE = """You are a wise, clever person with expertise in data analysis and psychology. You excel at analyzing text and behavioral data, gaining insights into the personal character, qualities, and hobbies of the authors of these texts. Additionally, you possess strong interpersonal skills, allowing you to communicate your insights clearly and effectively. You are an expert in analysis, with a specialization in psychology and data analysis. You can deeply understand text and behavioral data, using this information to gain insights into the author's character, qualities, and preferences. At the same time, you also have excellent communication skills, enabling you to share your observations and analysis results clearly and effectively.
"""

USR_MERGE = """
The user will provide you with multiple (>2) analysis contents regarding different areas of interest. 
However, we now consider these areas of interest to be quite similar or have the potential to be merged. 
Therefore, we need you to help merge these various analyzed interest domains. Your job is to identify the commonalities among these user interest analysis contents, extract a more general common interest domain, and then supplement relevant fields in this newly extracted common interest domain using the provided information from the original analyses.

Both the input user interest domain analysis contents and your output of the new common interest domain analysis result must follow this structure:
---
**[Name]**: {Interest Domain Name}  
**[Aspect]**: {Interest Domain Aspect}  
**[Icon]**: {The icon that best represents this interest}  
**[Description]**: {Brief description of the user's interests in this area}  
**[Content]**: {Detailed description of what activities the user has participated in or engaged with in this area, along with some analysis and reasoning}  
---
**[Timelines]**: {The development timeline of the user in this interest area, including dates, brief introductions, and referenced memory IDs}  
- {CreateTime}, {BriefDesc}, {refMemoryId}  
- xxxx  

You need to try to merge the interests into an appropriate new interest domain, and then write the corresponding analysis result from the perspective of this new field.

Shades:
{shades}

Your generated content should meet the following structure:
{
    "newInterestName": "xxx", 
    "newInterestAspect": "xxx", 
    "newInterestIcon": "xxx", 
    "newInterestDesc": "xxx", 
    "newInterestContent": "xxx", 
    "newInterestTimelines": [ 
        {
            "createTime": "xxx",
            "refMemoryId": xxx,
            "description": "xxx"
        },
        xxx
    ] 
}"""

SYS_IMPROVE = """You are a wise, clever person with expertise in data analysis and psychology. You excel at analyzing text and behavioral data, gaining insights into the personal character, qualities, and hobbies of the authors of these texts. Additionally, you possess strong interpersonal skills, allowing you to communicate your insights clearly and effectively. You are an expert in analysis, with a specialization in psychology and data analysis. You can deeply understand text and behavioral data, using this information to gain insights into the author's character, qualities, and preferences. At the same time, you also have excellent communication skills, enabling you to share your observations and analysis results clearly and effectively.

Now you need to help complete the following task:

The user will provide you a analysis result of a specific area of interest base on previous memories, with the structure as follows:
---
**[Name]**: {Interest Domain Name}
**[Aspect]**: {Interest Domain Aspect}
**[Icon]**: {The icon that best represents this interest}
**[Description]**: {Brief description of the user's interests in this area}
**[Content]**: {Detailed description of what activities the user has participated in or engaged with in this area, along with some analysis and reasoning}
---
**[Timelines]**  {The development timeline of the user in this interest area, including dates, brief introductions, and referenced memory IDs}
- {CreateTime}, {BriefDesc}, {refMemoryId}
- xxxx
"""

USR_IMPROVE = """
Now the user has recently added new memories. You need to appropriately update the previous analysis results based on these newly added memories and the previous memories. 

You need to follow these steps for modification:
1. First, determine whether the new memories are relevant to the current interest domain [based on the Pre-Version analysis results]. If none are relevant, you can skip the modification steps and ignore the rest.
2. If there are new memories related to the interest domain [based on the Pre-Version analysis results], then check the Description and Content fields whether update is necessary based on the new information in the memories and make corresponding additions to the Timeline section.
    2.1 Follow the sentence structure of the previous description. It should be a brief introduction that highlights the specific elements or topics referenced in the user's memory and should be in a single sentence. If the previous description can describe user's interest domain well, then updating the description is not necessary.
    2.2 The Content section can be relatively longer, so you can make appropriate adjustments to the Content based on the new memory information. If it's an entirely new part under this interest domain, you can supplement this content for the update. The modification length can be slightly longer than the Description section.
    2.3 For the Timeline section, follow the structure of the Pre-Version analysis results, and add the relevant memory timeline records.

Existing Shade Info:
{old_shade}

Recent Memories:
{new_memories}

You should generate follow format:
{
    "improveDesc": "xxx", # if no relevant new memories, this field should be None  
    "improveContent": "xxx", # if no relevant new memories, this field should be None  
    "improveTimelines": [ # if no relevant new memories, this field should be empty list
        {
            "createTime": "xxx",
            "refMemoryId": xxx,
            "description": "xxx"
        },
        xxx
    ] # For the improveTimeline field, you only need to add new timeline records for the new memory, and the existing timeline records are generated here.
}"""


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
        llm_service: LLMService,
        wasabi_adapter: WasabiStorageAdapter
    ):
        """
        Initialize the ShadeGenerator.
        
        Args:
            llm_service: Service for LLM interactions
            wasabi_adapter: Adapter for Wasabi storage operations
        """
        self.llm_service = llm_service
        self.wasabi_adapter = wasabi_adapter
    
        # Initialize parameters for LLM calls that match LPM Kernel
        self.preferred_language = "en"
        self.model_params = {
            "temperature": 0,
            "max_tokens": 3000,
            "top_p": 0,
            "frequency_penalty": 0,
            "seed": 42,
            "presence_penalty": 0,
            "timeout": 45,
        }
        self._top_p_adjusted = False  # Flag to track if top_p has been adjusted
    
    def _build_message(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """
        Builds the message structure for the LLM API, similar to LPM Kernel implementation.
        
        Args:
            system_prompt: The system prompt to guide the LLM behavior
            user_prompt: The user prompt containing the actual query
            
        Returns:
            A list of message dictionaries formatted for the LLM API
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Add preferred language instruction if specified
        if self.preferred_language:
            messages.append(
                {
                    "role": "system",
                    "content": f"Please respond in {self.preferred_language} language."
                }
            )
        
        return messages
        
    def __parse_json_response(
        self, content: str, pattern: str, default_res: dict = None
    ) -> Dict[str, Any]:
        """
        Parses JSON response from LLM output using regex pattern, similar to LPM Kernel.
        
        Args:
            content: The raw text response from the LLM
            pattern: Regex pattern to extract the JSON string
            default_res: Default result to return if parsing fails
            
        Returns:
            Parsed JSON dictionary or default_res if parsing fails
        """
        import re
        
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            logger.error(f"No JSON found: {content}")
            return default_res
            
        try:
            json_res = json.loads(matches[0])
        except Exception as e:
            logger.error(f"JSON parse error: {str(e)}")
            logger.error(f"Content: {content}")
            return default_res
            
        return json_res
    
    def generate_shade(
        self,
        user_id: str,
        old_memory_list: List[Note] = [],
        new_memory_list: List[Note] = [],
        shade_info_list: List[Dict[str, Any]] = []
    ) -> Optional[L1Shade]:
        """
        Generates or updates a shade based on memories.
        
        Implements three processing paths:
        1. Initial processing - creating a new shade from new_memory_list
        2. Merging then improving - merge multiple shade_info_list items then improve with new_memory_list
        3. Just improving - update a single shade_info_list item with new_memory_list
        
        Args:
            user_id: User ID
            old_memory_list: List of existing memories (compatibility parameter, not used)
            new_memory_list: List of new memories/notes to process
            shade_info_list: List of existing shade information
            
        Returns:
            Generated or updated shade, or None if generation failed
        """
        try:
            logger.info(f"Generate shade called with {len(shade_info_list)} shade_info_list and {len(new_memory_list)} new_memory_list")
            
            # Determine which processing path to take
            if not (shade_info_list or old_memory_list):
                # PATH 1: Initial shade processing
                logger.info(f"Shades initial Process! Current shade have {len(new_memory_list)} memories!")
                new_shade = self._initial_shade_process(new_memory_list)
            elif shade_info_list and old_memory_list:
                # PATH 2/3: Either merge multiple shades then improve, or just improve a single shade
                if len(shade_info_list) > 1:
                    # Multiple shades - merge them first
                    logger.info(f"Merge shades Process! {len(shade_info_list)} shades need to be merged!")
                    raw_shade = self._merge_shades_process(user_id, shade_info_list, old_memory_list)
                else:
                    # Single shade - use directly
                    raw_shade = shade_info_list[0]
                    
                # Improve the shade (either merged or single) with new memories
                logger.info(f"Update shade Process! Current shade should improve {len(new_memory_list)} memories!")
                new_shade = self._improve_shade_process(user_id, raw_shade, new_memory_list)
            else:
                # Abnormal input - either shade_info_list or old_memory_list is empty but not both
                logger.error("The shade_info_list or old_memory_list is empty! Please check the input!")
                raise Exception(
                    "The shade_info_list or old_memory_list is empty! Please check the input!"
                )
                
            # Check if new_shade is empty or None (focus on initial stage)
            if not new_shade:
                return None
            
            return new_shade
                
        except Exception as e:
            logger.error(f"Error in generate_shade: {str(e)}", exc_info=True)
            # Fallback to initial processing if anything fails
            if new_memory_list:
                return self._initial_shade_process(user_id, new_memory_list)
            return None
    
    def __add_second_view_info(self, shade: L1Shade) -> L1Shade:
        """
        Add second-person perspective information to the shade, similar to LPM Kernel.
        
        Args:
            shade: The L1Shade object with third-person perspective
            
        Returns:
            Updated L1Shade object with second-person perspective
        """
        if not shade.desc_third_view and not shade.content_third_view:
            # If no third-person view, use summary 
            shade.desc_third_view = shade.summary
            shade.content_third_view = shade.summary
            
        user_prompt = f"""Domain Name: {shade.name}
Domain Description: {shade.desc_third_view or shade.summary}
Domain Content: {shade.content_third_view or shade.summary}
Domain Timelines: 
{
    "-".join([f"{timeline.get('createTime', '')}, {timeline.get('description', '')}, {timeline.get('refMemoryId', timeline.get('refId', ''))}" 
             for timeline in shade.metadata.get("timelines", []) if timeline.get('isNew', True)])
}
"""
        # Generate second view info using a simplified person perspective shift prompt
        perspective_shift_prompt = """**Task:**
You will be provided with a comprehensive user analysis report with the following structure:

Domain Name: [Domain Name]
Domain Description: [Domain Description]
Domain Content: [Domain Content]
Domain Timelines: 
- [createTime], [description], [refMemoryId]
- xxxx

**Requirements:**
1. **Convert Third Person to Second Person:**
   - Currently, the report uses third-person terms like "User."
   - Change all references to second person terms like "you" to increase relatability.

2. **Modify Descriptions:**
   - Adjust all descriptions in the **Domain Description**, **Domain Content**, and **Timeline description** sections to reflect the second person perspective.

3. **Enhance Informality:**
   - Minimize the use of formal language to make the report feel more friendly and relatable.

**Response Format:**
{
    "domainName": str (keep the same with the original),
    "domainDesc": str (modify to second person perspective),
    "domainContent": str (modify to second person perspective),
    "domainTimeline": [
        {
            "createTime": str (keep the same with the original),
            "refMemoryId": int (keep the same with the original),
            "description": str (modify to second person perspective)
        },
        ...
    ]
}"""
        perspective_shift_message = self._build_message(perspective_shift_prompt, user_prompt)
        
        try:
            response = self.llm_service.call_with_retry(perspective_shift_message, model_params=self.model_params)
            content = response.choices[0].message.content
            
            # Parse result
            shift_pattern = r"\{.*\}"
            shift_perspective_result = self.__parse_json_response(content, shift_pattern)
            
            if shift_perspective_result:
                # Add second view info to shade
                shade.add_second_view(
                    domain_desc=shift_perspective_result.get("domainDesc", ""),
                    domain_content=shift_perspective_result.get("domainContent", ""),
                    domain_timeline=shift_perspective_result.get("domainTimeline", [])
                )
                
        except Exception as e:
            logger.error(f"Error adding second view to shade: {str(e)}", exc_info=True)
            # If failed, use third-person view as fallback
            shade.desc_second_view = shade.desc_third_view
            shade.content_second_view = shade.content_third_view
            
        return shade

    def __shade_initial_postprocess(self, content: str, user_id: str, notes: List[Note]) -> Optional[L1Shade]:
        """
        Processes the initial shade generation response, similar to LPM Kernel.
        
        Args:
            content: Raw LLM response text
            user_id: User ID for creating the shade
            notes: Original notes used to generate the shade
            
        Returns:
            L1Shade object or None if processing fails
        """
        # Use the same pattern as LPM Kernel
        shade_generate_pattern = r"\{.*\}"
        shade_raw_info = self.__parse_json_response(content, shade_generate_pattern)

        if not shade_raw_info:
            logger.error(f"Failed to parse the shade generate result: {content}")
            return None
                
        logger.info(f"Shade Generate Result: {shade_raw_info}")
        
        # Extract data mapping LPM Kernel fields to ours
        name = shade_raw_info.get("domainName", "")
        if not name and "name" in shade_raw_info:  # Fallback to our naming
            name = shade_raw_info.get("name", "")
            
        # Get aspect and icon
        aspect = shade_raw_info.get("aspect", "")
        icon = shade_raw_info.get("icon", "")
            
        # Extract the descriptions and content
        desc_third_view = shade_raw_info.get("domainDesc", "")
        content_third_view = shade_raw_info.get("domainContent", "")
        
        # For compatibility with our existing format
        summary = content_third_view
        if not summary and "summary" in shade_raw_info:
            summary = shade_raw_info.get("summary", "")
            
        # Extract confidence differently depending on format
        confidence = 0.0
        if "confidence" in shade_raw_info:
            confidence = shade_raw_info.get("confidence", 0.0)
            
        # Get timelines from either format
        timelines = []
        if "domainTimelines" in shade_raw_info:
            timelines = shade_raw_info.get("domainTimelines", [])
        elif "timelines" in shade_raw_info:
            timelines = shade_raw_info.get("timelines", [])
            
        # Store initial shade data in Wasabi
        # TODO: don't do this here yet?
        s3_path = self._store_shade_data(user_id, {
            "name": name,
            "summary": summary,
            "confidence": confidence,
            "timelines": timelines,
            "aspect": aspect,
            "icon": icon,
            "desc_third_view": desc_third_view,
            "content_third_view": content_third_view
        }, notes)
        
        # Prepare metadata with timelines
        metadata = {
            "cluster_size": len(notes),
            "timelines": timelines
        }
        
        # Create shade object with all the fields
        shade_kwargs = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": name,
            "summary": summary,
            "confidence": confidence,
            "metadata": metadata,
            "aspect": aspect,
            "icon": icon,
            "desc_third_view": desc_third_view,
            "content_third_view": content_third_view,
        }
        
        # Only add s3_path if the class accepts it
        import inspect
        if 's3_path' in inspect.signature(L1Shade.__init__).parameters:
            shade_kwargs["s3_path"] = s3_path
            
        # Create shade object
        shade = L1Shade(**shade_kwargs)
        
        # Add second-person view information
        shade = self.__add_second_view_info(shade)
        
        logger.info(f"Generated shade: {shade.name} with confidence {shade.confidence}")
        return shade
    
    def _initial_shade_process(
        self,
        user_id: str,
        notes: List[Note]
    ) -> Optional[L1Shade]:
        """
        Process the initial shade generation from new notes, matching LPM Kernel's approach.
        
        Args:
            user_id: User ID
            notes: List of notes to process
            
        Returns:
            New shade or None if generation failed
        """
        try:
            if not notes or len(notes) == 0:
                logger.warning("No notes provided for shade generation")
                return None
            
            # Format documents for prompt using Note's to_str method
            user_prompt = "\n\n".join([note.to_str() for note in notes])
            
            # Build message like LPM Kernel
            shade_generate_message = self._build_message(SYS_SHADE, USR_SHADE.format(documents=user_prompt))
            
            logger.info(f"Generating shade for {len(notes)} notes")
            # Use call_with_retry with our model_params like LPM Kernel
            response = self.llm_service.call_with_retry(shade_generate_message, model_params=self.model_params)
            content = response.choices[0].message.content
            
            logger.info(f"Shade Generate Result: {content}")
            # Use our postprocessing method
            return self.__shade_initial_postprocess(content, user_id, notes)
            
        except Exception as e:
            logger.error(f"Error in initial shade process: {str(e)}", exc_info=True)
            return None

    def _format_new_notes_for_improvement(self, new_notes: List[Note]) -> str:
        """
        Format new notes to include in improvement prompt, matching LPM Kernel's expected format.
        
        Args:
            new_notes: List of new notes to format
            
        Returns:
            Formatted notes text
        """
        if not new_notes:
            return ""
        
        formatted_text = "**New Memories**:\n\n"
        formatted_text += "\n\n".join([note.to_str() for note in new_notes])
        
        return formatted_text
    
    def __shade_merge_postprocess(self, content: str, user_id: str, shade_objects: List[L1Shade]) -> Optional[L1Shade]:
        """
        Processes the shade merging response, similar to LPM Kernel.
        
        Args:
            content: Raw LLM response text
            user_id: User ID for creating the merged shade
            shade_objects: Original shades used for merging
            
        Returns:
            Merged L1Shade object or None if processing fails
        """
        # Use the same pattern as LPM Kernel
        shade_merge_pattern = r"\{.*\}"
        merged_shade_info = self.__parse_json_response(content, shade_merge_pattern)
        
        if not merged_shade_info:
            logger.error(f"Failed to parse the shade merge result: {content}")
            return None if len(shade_objects) > 1 else shade_objects[0]
        
        logger.info(f"Shade Merge Result: {merged_shade_info}")
        
        # Extract data using LPM Kernel field names with fallbacks to our naming
        name = merged_shade_info.get("newInterestName", "")
        if not name and "name" in merged_shade_info:
            name = merged_shade_info.get("name", "Merged Shade")
        
        # Get aspect and icon
        aspect = merged_shade_info.get("newInterestAspect", "")
        icon = merged_shade_info.get("newInterestIcon", "")
        
        # Extract the descriptions and content
        desc_third_view = merged_shade_info.get("newInterestDesc", "")
        content_third_view = merged_shade_info.get("newInterestContent", "")
        
        # For compatibility, use content_third_view as summary if available
        summary = content_third_view
        if not summary and "summary" in merged_shade_info:
            summary = merged_shade_info.get("summary", "")
        
        confidence = 0.0
        if "confidence" in merged_shade_info:
            confidence = merged_shade_info.get("confidence", 0.0)
        
        # Get timelines from either format
        timelines = []
        if "newInterestTimelines" in merged_shade_info:
            timelines = merged_shade_info.get("newInterestTimelines", [])
        elif "timelines" in merged_shade_info:
            timelines = merged_shade_info.get("timelines", [])
        
        # Calculate center embedding if available
        center_embedding = self._calculate_merged_center_embedding(shade_objects)
                
        # Gather all timelines from source shades
        all_timelines = []
        for shade in shade_objects:
            if hasattr(shade, "metadata") and "timelines" in shade.metadata:
                all_timelines.extend(shade.metadata["timelines"])
                
        # Add any new timelines from the LLM response
        all_timelines.extend(timelines)
                
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
        
        # Store the merged shade data in Wasabi
        merged_shade_data = {
            "name": name,
            "summary": summary,
            "confidence": confidence,
            "timelines": sorted_timelines,
            "aspect": aspect,
            "icon": icon,
            "desc_third_view": desc_third_view,
            "content_third_view": content_third_view
        }
        
        if center_embedding is not None:
            merged_shade_data["center_embedding"] = center_embedding
        
        # TODO: do we do this here?
        s3_path = self._store_merged_shade_data(user_id, merged_shade_data)
        
        # Create the merged shade object
        metadata = {
            "source_shades": [s.id for s in shade_objects],
            "timelines": sorted_timelines
        }
        
        if center_embedding is not None:
            metadata["center_embedding"] = center_embedding
        
        shade_kwargs = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": name,
            "summary": summary,
            "confidence": confidence,
            "metadata": metadata,
            "aspect": aspect,
            "icon": icon,
            "desc_third_view": desc_third_view, 
            "content_third_view": content_third_view
        }
        
        # Only add s3_path if it's used in the model
        import inspect
        if 's3_path' in inspect.signature(L1Shade.__init__).parameters:
            shade_kwargs["s3_path"] = s3_path
        
        # Create the merged shade object
        merged_shade = L1Shade(**shade_kwargs)
        
        # Add second-person view information
        merged_shade = self.__add_second_view_info(merged_shade)
        
        logger.info(f"Merged shade: {merged_shade.name} with confidence {merged_shade.confidence}")
        return merged_shade

    def _merge_shades_process(
        self,
        user_id: str,
        shades: List[Dict[str, Any]],
        old_memory_list: List[Note] = []
    ) -> Optional[L1Shade]:
        """
        Process the merging of multiple shades, matching LPM Kernel's approach.
        
        Args:
            user_id: User ID
            shades: List of shades to merge
            old_memory_list: List of old memories (optional, for compatibility with lpm_kernel)
            
        Returns:
            Merged shade or None if merging failed
        """
        try:
            # Convert dictionaries to L1Shade objects if needed
            shade_objects = []
            for shade in shades:
                if isinstance(shade, dict):
                    # Convert dict to L1Shade object if needed
                    if 'id' in shade and 'name' in shade and 'summary' in shade:
                        shade_objects.append(L1Shade(**shade))
                    else:
                        logger.warning(f"Invalid shade dictionary: {shade}")
                else:
                    shade_objects.append(shade)
            
            if not shade_objects or len(shade_objects) == 0:
                logger.warning("No valid shades for merging")
                return None
            
            # Format shades for prompt, matching lpm_kernel's approach
            shades_text = self._format_shades_for_prompt(shade_objects)
            
            # Generate merged shade using LLM, matching lpm_kernel's approach
            merge_shades_message = self._build_message(SYS_MERGE, USR_MERGE.format(shades=shades_text))
            
            logger.info(f"Merging {len(shade_objects)} shades")
            response = self.llm_service.call_with_retry(merge_shades_message, model_params=self.model_params)
            content = response.choices[0].message.content
            
            logger.info(f"Shade Merge Result: {content}")
            # Process the response using our merge post-processing
            return self.__shade_merge_postprocess(content, user_id, shade_objects)
            
        except Exception as e:
            logger.error(f"Error in merge shades process: {str(e)}", exc_info=True)
            return None if len(shades) > 1 else shade_objects[0]
            
    def __shade_improve_postprocess(self, content: str, user_id: str, old_shade: L1Shade, new_notes: List[Note]) -> Optional[L1Shade]:
        """
        Processes the shade improvement response, similar to LPM Kernel.
        
        Args:
            content: Raw LLM response text
            user_id: User ID for the shade
            old_shade: Original shade being improved
            new_notes: New notes used for improvement
            
        Returns:
            Improved L1Shade object or the original shade if processing fails
        """
        # Use the same pattern as LPM Kernel
        shade_improve_pattern = r"\{.*\}"
        improved_data = self.__parse_json_response(content, shade_improve_pattern)
        
        if not improved_data:
            logger.error(f"Failed to parse improved shade response: {content}")
            return old_shade
        
        logger.info(f"Shade Improve Result: {improved_data}")
        
        # Extract improved data using LPM Kernel field names specifically
        improved_name = old_shade.name
        
        # Get descriptions and content updates - prioritize LPM Kernel naming
        improved_desc_third_view = old_shade.desc_third_view
        if "improveDesc" in improved_data and improved_data["improveDesc"] is not None:
            improved_desc_third_view = improved_data.get("improveDesc")
        
        improved_content_third_view = old_shade.content_third_view
        if "improveContent" in improved_data and improved_data["improveContent"] is not None:
            improved_content_third_view = improved_data.get("improveContent")
        
        # Update summary from content if available
        improved_summary = improved_content_third_view
        
        improved_confidence = old_shade.confidence
        
        # Get new timelines - prioritize LPM Kernel naming
        new_timelines = []
        if "improveTimelines" in improved_data:
            new_timelines = improved_data.get("improveTimelines", [])
        
        # Update the shade with improved data
        updated_shade_data = {
            "id": old_shade.id,
            "user_id": user_id,
            "name": improved_name,
            "summary": improved_summary,
            "confidence": improved_confidence,
            "metadata": old_shade.metadata.copy() if hasattr(old_shade, "metadata") else {},
            "aspect": old_shade.aspect,
            "icon": old_shade.icon,
            "desc_third_view": improved_desc_third_view,
            "content_third_view": improved_content_third_view,
            "desc_second_view": old_shade.desc_second_view,
            "content_second_view": old_shade.content_second_view
        }
        
        # Add s3_path if it exists in the original shade
        if hasattr(old_shade, "s3_path") and old_shade.s3_path:
            updated_shade_data["s3_path"] = old_shade.s3_path
        
        # Add new timelines to metadata if available
        if "metadata" in updated_shade_data and "timelines" in updated_shade_data["metadata"]:
            existing_timelines = updated_shade_data["metadata"]["timelines"]
        else:
            if "metadata" not in updated_shade_data:
                updated_shade_data["metadata"] = {}
            existing_timelines = []
        
        updated_shade_data["metadata"]["timelines"] = existing_timelines + new_timelines
        
        # Create new shade object
        improved_shade = L1Shade(**updated_shade_data)
        
        # Store improved shade data if significant changes made
        # TODO: don't do this here yet?
        if (improved_summary != old_shade.summary or 
            improved_desc_third_view != old_shade.desc_third_view or 
            improved_content_third_view != old_shade.content_third_view or
            new_timelines):
            improved_shade_dict = {
                "name": improved_shade.name,
                "summary": improved_shade.summary,
                "confidence": improved_shade.confidence,
                "timelines": updated_shade_data["metadata"].get("timelines", []),
                "aspect": improved_shade.aspect,
                "icon": improved_shade.icon,
                "desc_third_view": improved_shade.desc_third_view,
                "content_third_view": improved_shade.content_third_view
            }
            s3_path = self._store_improved_shade_data(user_id, improved_shade_dict, new_notes)
            
            # Update s3_path if the model accepts it
            if hasattr(improved_shade, "s3_path"):
                improved_shade.s3_path = s3_path
        
        # Update second-person views if third-person views changed
        if (improved_desc_third_view != old_shade.desc_third_view or 
            improved_content_third_view != old_shade.content_third_view):
            improved_shade = self.__add_second_view_info(improved_shade)
        
        logger.info(f"Improved shade: {improved_shade.name}")
        return improved_shade
    
    def _improve_shade_process(
        self,
        user_id: str,
        old_shade: Union[Dict[str, Any], L1Shade],
        new_notes: List[Note]
    ) -> Optional[L1Shade]:
        """
        Process the improvement of a shade with new notes, matching LPM Kernel's approach.
        
        Args:
            user_id: User ID
            old_shade: Existing shade to improve
            new_notes: New notes to incorporate
            
        Returns:
            Improved shade or None if improvement failed
        """
        try:
            if not new_notes or len(new_notes) == 0:
                logger.warning("No new notes provided for shade improvement")
                # Return original shade if no new notes
                return old_shade if isinstance(old_shade, L1Shade) else L1Shade(**old_shade)
            
            # Convert dictionary to L1Shade if needed
            if isinstance(old_shade, dict):
                if 'id' in old_shade and 'name' in old_shade and 'summary' in old_shade:
                    old_shade = L1Shade(**old_shade)
                else:
                    logger.warning(f"Invalid old_shade dictionary: {old_shade}")
                    return self._initial_shade_process(user_id, new_notes)
            
            # Format the existing shade info and new notes for prompt
            old_shade_text = self._format_shade_for_improvement(old_shade)
            new_notes_text = self._format_new_notes_for_improvement(new_notes)
            
            # Generate improved shade using LLM
            # Use _build_message like LPM Kernel
            shade_improve_message = self._build_message(
                SYS_IMPROVE, 
                USR_IMPROVE.format(old_shade=old_shade_text, new_memories=new_notes_text)
            )
            
            logger.info(f"Improving shade with {len(new_notes)} new notes")
            # Use call_with_retry with model_params
            response = self.llm_service.call_with_retry(shade_improve_message, model_params=self.model_params)
            content = response.choices[0].message.content
            
            logger.info(f"Shade Improve Result: {content}")
            # Use our improvement post-processing
            return self.__shade_improve_postprocess(content, user_id, old_shade, new_notes)
            
        except Exception as e:
            logger.error(f"Error improving shade: {str(e)}", exc_info=True)
            return old_shade if isinstance(old_shade, L1Shade) else None
    
    def _format_shades_for_prompt(self, shades: List[L1Shade]) -> str:
        """
        Format shades for inclusion in LLM prompt, matching lpm_kernel's approach.
        
        Args:
            shades: List of shades to format
            
        Returns:
            Formatted shades text
        """
        return "\n\n".join(
            [
                f"User Interest Domain {i+1} Analysis:\n{shade.to_str()}"
                for i, shade in enumerate(shades)
            ]
        )
    
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
    
    def _format_shade_for_improvement(self, shade: L1Shade) -> str:
        """
        Format a shade for inclusion in improvement prompt, matching LPM Kernel's expected format.
        
        Args:
            shade: Shade to format
            
        Returns:
            Formatted shade text
        """
        # Format timelines
        timelines_text = ""
        if hasattr(shade, "timelines") and shade.timelines:
            for timeline in shade.timelines:
                timelines_text += f"- {timeline.createTime}, {timeline.descThirdView}, {timeline.refMemoryId}\n"
        elif hasattr(shade, "metadata") and "timelines" in shade.metadata:
            for timeline in shade.metadata["timelines"]:
                create_time = timeline.get("createTime", "")
                description = timeline.get("description", "")
                ref_id = timeline.get("refId", timeline.get("refMemoryId", ""))
                timelines_text += f"- {create_time}, {description}, {ref_id}\n"
        
        # Format the shade in the exact structure expected by LPM Kernel
        formatted_text = f"""---
**[Name]**: {shade.name}
**[Aspect]**: {shade.aspect}
**[Icon]**: {shade.icon}
**[Description]**: 
{shade.desc_third_view or shade.summary}

**[Content]**: 
{shade.content_third_view or shade.summary}
---

**[Timelines]**:
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