"""
ShadeMerger for merging similar shades.

This module provides the ShadeMerger class that merges similar shades together.
It acts as a wrapper around ShadeGenerator's merge_shades method to provide a
compatible API with lpm_kernel's ShadeMerger class.
"""
import logging
from typing import List, Dict, Any, Optional

from app.models.l1.shade import L1Shade, MergedShadeResult
from app.processors.l1.shade_generator import ShadeGenerator

logger = logging.getLogger(__name__)

class MergeShadeResult:
    """Result of merging shades."""
    def __init__(self, success: bool, merge_shade_list: List[Dict[str, Any]]):
        self.success = success
        self.merge_shade_list = merge_shade_list

class ShadeMerger:
    """
    Merges similar shades together.
    
    This class is a wrapper around ShadeGenerator's merge_shades method to
    provide a compatible API with lpm_kernel's ShadeMerger class.
    
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
    
    def merge_shades(self, user_id: str, shades: List[L1Shade]) -> MergedShadeResult:
        """
        Merge multiple shades into a single shade, using the ShadeGenerator.
        
        Args:
            user_id: User ID
            shades: List of shades to merge
            
        Returns:
            MergedShadeResult indicating success or failure
        """
        try:
            # Merge shades using the ShadeGenerator
            # Note: ShadeGenerator does not have a merge_shades method, but has _merge_shades_process
            merged_shade = self.shade_generator._merge_shades_process(user_id, shades)
            
            if merged_shade:
                return MergedShadeResult(
                    success=True,
                    merge_shade_list=[merged_shade.to_dict()]
                )
            else:
                logger.error("Failed to merge shades")
                return MergedShadeResult(
                    success=False,
                    merge_shade_list=[],
                    error="Failed to merge shades"
                )
                
        except Exception as e:
            logger.error(f"Error merging shades: {str(e)}", exc_info=True)
            return MergedShadeResult(
                success=False,
                merge_shade_list=[],
                error=f"Error merging shades: {str(e)}"
            ) 