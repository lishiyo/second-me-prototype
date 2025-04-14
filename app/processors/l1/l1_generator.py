"""
L1Generator for generating L1 level knowledge representations.

This module provides the L1Generator class that contains the core logic for
generating L1 level knowledge representations. It delegates to specialized
generators for topics, shades, and biographies.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from app.models.l1.note import Note
from app.models.l1.bio import Bio
from app.models.l1.shade import Shade as L1Shade
from app.processors.l1.topics_generator import TopicsGenerator
from app.processors.l1.shade_generator import ShadeGenerator
from app.processors.l1.biography_generator import BiographyGenerator

logger = logging.getLogger(__name__)

class MergeShadeResult:
    """Result of merging shades."""
    def __init__(self, success: bool, merge_shade_list: List[Dict[str, Any]]):
        self.success = success
        self.merge_shade_list = merge_shade_list


class L1Generator:
    """
    Core logic for generating L1 representations.
    
    This class delegates to specialized generators (TopicsGenerator, ShadeGenerator,
    BiographyGenerator) and provides utility functions for generating L1 data.
    
    Attributes:
        topics_generator: Generator for topics and clusters
        shade_generator: Generator for shades
        biography_generator: Generator for biographies
    """
    
    def __init__(
        self,
        topics_generator: Optional[TopicsGenerator] = None,
        shade_generator: Optional[ShadeGenerator] = None,
        biography_generator: Optional[BiographyGenerator] = None
    ):
        """
        Initialize the L1Generator.
        
        Args:
            topics_generator: TopicsGenerator instance
            shade_generator: ShadeGenerator instance
            biography_generator: BiographyGenerator instance
        """
        self.topics_generator = topics_generator or TopicsGenerator()
        self.shade_generator = shade_generator or ShadeGenerator()
        self.biography_generator = biography_generator or BiographyGenerator()
    
    def generate_topics(self, notes_list: List[Note]) -> Dict[str, Any]:
        """
        Generate topics from a list of notes.
        
        Args:
            notes_list: List of notes to generate topics from
            
        Returns:
            Generated topics data
        """
        return self.topics_generator.generate_topics(notes_list)
    
    def gen_topics_for_shades(
        self,
        user_id: str,
        old_cluster_list: List[Dict[str, Any]],
        old_outlier_memory_list: List[Dict[str, Any]],
        new_memory_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate topic clusters for shades.
        
        Args:
            user_id: User ID
            old_cluster_list: List of existing clusters
            old_outlier_memory_list: List of outlier memories from previous run
            new_memory_list: List of new memories to process
            
        Returns:
            Dictionary containing updated cluster list and outlier memory list
        """
        return self.topics_generator.generate_topics_for_shades(
            old_cluster_list,
            old_outlier_memory_list,
            new_memory_list
        )
    
    def gen_shade_for_cluster(
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
        return self.shade_generator.generate_shade_for_cluster(
            user_id,
            old_shades,
            cluster_notes,
            memory_list
        )
    
    def merge_shades(
        self,
        user_id: str,
        shades: List[L1Shade]
    ) -> MergeShadeResult:
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
            
            merged_shades = self.shade_generator.merge_shades(user_id, shades)
            return MergeShadeResult(
                success=True,
                merge_shade_list=merged_shades
            )
        except Exception as e:
            logger.error(f"Error merging shades: {str(e)}", exc_info=True)
            return MergeShadeResult(success=False, merge_shade_list=[])
    
    def gen_global_biography(
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
        return self.biography_generator.generate_global_biography(
            user_id,
            old_profile,
            cluster_list
        )
    
    def gen_status_biography(
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
        return self.biography_generator.generate_status_biography(
            user_id,
            recent_documents,
            old_bio
        ) 