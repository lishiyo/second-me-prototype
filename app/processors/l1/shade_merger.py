"""
ShadeMerger for merging similar shades.

This module provides the ShadeMerger class that merges similar shades together.
It acts as a wrapper around ShadeGenerator's merge_shades method to provide a
compatible API with lpm_kernel's ShadeMerger class.
"""
import logging
from typing import List, Dict, Any, Optional

from app.models.l1.shade import Shade as L1Shade
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
    
    def merge_shades(self, user_id: str, shades: List[L1Shade]) -> MergeShadeResult:
        """
        Merge multiple shades into a coherent representation.
        
        Args:
            user_id: User ID
            shades: List of shades to merge
            
        Returns:
            MergeShadeResult containing success status and merged shades
        """
        try:
            if not shades or len(shades) == 0:
                logger.warning("No shades to merge")
                return MergeShadeResult(success=False, merge_shade_list=[])
            
            # Single shade doesn't need merging
            if len(shades) == 1:
                shade_dict = shades[0].to_dict() if hasattr(shades[0], 'to_dict') else shades[0]
                return MergeShadeResult(success=True, merge_shade_list=[shade_dict])
            
            # Delegate to ShadeGenerator.merge_shades
            merged_shades = self.shade_generator.merge_shades(user_id, shades)
            
            # Ensure metadata with timelines is included
            for shade in merged_shades:
                if "metadata" not in shade:
                    shade["metadata"] = {}
                if "timelines" not in shade["metadata"] and "timelines" in shade:
                    shade["metadata"]["timelines"] = shade.get("timelines", [])
            
            return MergeShadeResult(
                success=True,
                merge_shade_list=merged_shades
            )
        except Exception as e:
            logger.error(f"Error merging shades: {str(e)}", exc_info=True)
            return MergeShadeResult(success=False, merge_shade_list=[]) 