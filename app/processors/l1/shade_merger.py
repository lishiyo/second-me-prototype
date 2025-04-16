"""
ShadeMerger for merging similar shades.

This module provides the ShadeMerger class that merges similar shades together.
It acts as a wrapper around ShadeGenerator's merge_shades method to provide a
compatible API with lpm_kernel's ShadeMerger class.
"""
import logging
import re
import json
from typing import List, Dict, Any, Optional
import numpy as np

from app.models.l1.shade import L1Shade, MergedShadeResult, ShadeMergeInfo
from app.processors.l1.shade_generator import ShadeGenerator

logger = logging.getLogger(__name__)

# Import the prompt from lpm_kernel (same as in ShadeGenerator)
SHADE_MERGE_DEFAULT_SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing and merging similar user identity shades. Your task involves three steps:

1. First, analyze each shade's core characteristics based on its:
   - Name
   - Aspect
   - Description (Third View)
   - Content (Third View)

2. Then, identify which shades can be merged by:
   - Looking for semantic similarities in core characteristics
   - Identify shades that can be turned into more complete content when merged 
   - Finding overlapping interests or behaviors
   - Identifying complementary traits
   - Evaluating the context and meaning

3. Finally, output mergeable shade groups where:
   - Each shade can only appear in one merge group
   - Multiple merge groups are allowed
   - Each merge group must contain at least 2 shades
   - If no shades need to be merged, return an empty array []

Your output must be a JSON array of arrays, where each inner array contains the IDs of shades that can be merged. For example:
[
    ["shade_id1", "shade_id2"],
    ["shade_id3", "shade_id4", "shade_id5"],
    ["shade_id6", "shade_id7"]
]

Or if no shades need to be merged:
[]

Important:
- Only output the JSON array, no additional text
- Ensure each shade ID appears only once across all groups
- Each group must contain at least 2 shade IDs
- Only suggest merging when there is strong evidence of similarity or redundancy"""

class ShadeMerger:
    """
    Merges similar shades together.
    
    This class follows lpm_kernel's ShadeMerger implementation for determining
    which shades can be merged together based on similarity.
    
    Attributes:
        shade_generator: ShadeGenerator instance for delegating merge operations
    """
    
    def __init__(self, shade_generator: ShadeGenerator):
        """
        Initialize the ShadeMerger.
        
        Args:
            shade_generator: ShadeGenerator instance for delegating merge operations
        """
        self.shade_generator = shade_generator
    
    def _build_user_prompt(self, shades: List[L1Shade]) -> str:
        """
        Builds a user prompt from shade list, matching lpm_kernel's implementation.
        
        Args:
            shades: List of shades to format for the prompt
            
        Returns:
            Formatted string containing shade information
        """
        shades_str = "\n\n".join(
            [
                f"Shade ID: {shade.id}\n"
                f"Name: {shade.name}\n"
                f"Aspect: {shade.aspect}\n"
                f"Description Third View: {shade.desc_third_view}\n"
                f"Content Third View: {shade.content_third_view}\n"
                for shade in shades
            ]
        )

        return f"""Shades List:
{shades_str}
"""
    
    def _parse_json_response(self, content: str, pattern: str, default_res=None) -> Any:
        """
        Parses JSON response from LLM output, matching lpm_kernel implementation.
        
        Args:
            content: The raw text response from the LLM
            pattern: Regex pattern to extract the JSON string
            default_res: Default result to return if parsing fails
            
        Returns:
            Parsed JSON object or default_res if parsing fails
        """
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
    
    def _calculate_merged_shades_center_embed(self, shades: List[L1Shade]) -> List[float]:
        """
        Calculates the center embedding for merged shades, matching lpm_kernel's implementation.
        
        Args:
            shades: List of shades to merge
            
        Returns:
            A list of floats representing the new center embedding
            
        Raises:
            ValueError: If no valid shades found or total cluster size is zero
        """
        if not shades:
            raise ValueError("No valid shades found for the given merge list.")

        # Find embedding dimension from first shade with an embedding
        shades_with_embedding = [s for s in shades if "center_embedding" in s.metadata and s.metadata["center_embedding"]]
        if not shades_with_embedding:
            raise ValueError("No shades with center_embedding found.")
            
        # Get embedding dimension
        first_embedding = shades_with_embedding[0].metadata["center_embedding"]
        embedding_dim = len(first_embedding)
            
        # Initialize calculation
        total_embedding = np.zeros(embedding_dim)
        total_cluster_size = 0

        for shade in shades_with_embedding:
            # Get cluster size (default to 1 if not specified)
            cluster_size = shade.metadata.get("cluster_size", 1)
            center_embedding = np.array(shade.metadata["center_embedding"])
            
            # Add weighted embedding
            total_embedding += cluster_size * center_embedding
            total_cluster_size += cluster_size

        if total_cluster_size == 0:
            raise ValueError("Total cluster size is zero, cannot compute the new center embedding.")

        # Calculate the weighted average
        new_center_embedding = total_embedding / total_cluster_size
        return new_center_embedding.tolist()
    
    def merge_shades(self, user_id: str, shades: List[L1Shade], test_mode: bool = False) -> MergedShadeResult:
        """
        Identify groups of similar shades that can be merged, following lpm_kernel's approach exactly.
        
        Args:
            user_id: User ID
            shades: List of shades to evaluate for merging
            test_mode: If True, force all shades to be merged (for testing purposes)
            
        Returns:
            MergedShadeResult with groups of shade IDs that can be merged and their center embeddings
        """
        try:
            if not shades or len(shades) < 2:
                logger.info("Not enough shades to merge (need at least 2)")
                return MergedShadeResult(
                    success=True,
                    merge_shade_list=[]
                )
            
            # Build the prompt with shade information
            user_prompt = self._build_user_prompt(shades)
            merge_decision_message = self.shade_generator._build_message(
                SHADE_MERGE_DEFAULT_SYSTEM_PROMPT, user_prompt
            )
            
            logger.info(f"Analyzing {len(shades)} shades for potential merges")
            
            # Either get merge decisions from LLM or force test merge
            merge_shade_list = []
            if test_mode:
                logger.info("Test mode: forcing shades to merge")
                # In test mode, merge all shades together
                merge_shade_list = [[str(shade.id) for shade in shades]]
            else:
                # Normal mode: Call LLM to get merge decisions
                response = self.shade_generator.llm_service.call_with_retry(
                    merge_decision_message,
                    model_params=self.shade_generator.model_params
                )
                content = response.choices[0].message.content
                logger.info(f"Shade Merge Decision Result: {content}")
                
                # Parse the response to get merge groups
                merge_shade_list = self._parse_json_response(content, r"\[.*\]", [])
            
            logger.info(f"Parsed merge groups: {merge_shade_list}")
            
            if not merge_shade_list:
                logger.info("No shades need to be merged")
                return MergedShadeResult(
                    success=True,
                    merge_shade_list=[]
                )
            
            # Process each merge group - this is exactly what lpm_kernel does
            final_merge_shade_list = []
            for group in merge_shade_list:
                shade_ids = group  # Each group is a list of shade IDs
                if not shade_ids or len(shade_ids) < 2:
                    logger.info(f"Skipping invalid merge group: {shade_ids}")
                    continue
                
                # Fetch shades based on shade IDs
                group_shades = [shade for shade in shades if str(shade.id) in shade_ids]
                
                # Skip if we don't have at least 2 valid shades
                if len(group_shades) < 2:
                    logger.info(f"Not enough valid shades for group {shade_ids}")
                    continue
                
                try:
                    # Calculate the new cluster embedding (center vector)
                    new_cluster_embedd = self._calculate_merged_shades_center_embed(group_shades)
                    logger.info(f"Calculated new cluster embedding for group {shade_ids}")
                    
                    # Add to final list without merging - just like lpm_kernel
                    final_merge_shade_list.append({
                        "shadeIds": shade_ids,
                        "centerEmbedding": new_cluster_embedd
                    })
                except ValueError as e:
                    logger.error(f"Error calculating center embedding for group {shade_ids}: {str(e)}")
                    # Skip this group if we can't calculate center embedding
                    continue
            
            # Return success with the merge information, not the merged shades
            return MergedShadeResult(
                success=True,
                merge_shade_list=final_merge_shade_list
            )
                
        except Exception as e:
            logger.error(f"Error in merge_shades: {str(e)}", exc_info=True)
            return MergedShadeResult(
                success=False,
                merge_shade_list=[],
                error=f"Error merging shades: {str(e)}"
            ) 