"""
Examples demonstrating how to use the L1 models, especially L1Shade.

This module provides practical examples for:
1. Creating and using L1Shade objects
2. Converting between different shade models
3. Working with ShadeTimeline objects
4. Interacting with the database
"""

from typing import Dict, Any, List
from datetime import datetime
import json

from app.models.l1.shade import L1Shade, ShadeInfo, ShadeMergeInfo, MergedShadeResult, ShadeTimeline
from app.models.l1.db_models import L1Shade as DBL1Shade
from app.providers.rel_db import RelationalDB

# Example 1: Creating a basic L1Shade
def create_basic_l1_shade():
    """Example of creating a basic L1Shade object"""
    # Create with minimum required fields
    shade = L1Shade(
        id="12345",
        user_id="user123",
        name="Programming Languages",
        summary="An overview of various programming languages and their applications."
    )
    
    # Create with all fields, including LPM Kernel compatible fields
    detailed_shade = L1Shade(
        id="67890",
        user_id="user123",
        name="Machine Learning",
        summary="An overview of machine learning techniques",
        confidence=0.95,
        aspect="Technology",
        icon="🧠",
        desc_third_view="The user has extensive knowledge about machine learning algorithms and applications.",
        content_third_view="The user has studied machine learning for several years, with particular focus on neural networks and deep learning architectures. They have completed several projects in natural language processing and computer vision.",
        desc_second_view="You have extensive knowledge about machine learning algorithms and applications.",
        content_second_view="You have studied machine learning for several years, with particular focus on neural networks and deep learning architectures. You have completed several projects in natural language processing and computer vision.",
        metadata={
            "timelines": [
                {
                    "createTime": "2022-01-15",
                    "description": "Completed a deep learning course",
                    "refMemoryId": "doc123"
                },
                {
                    "createTime": "2022-06-20",
                    "description": "Built a computer vision project",
                    "refMemoryId": "doc456"
                }
            ]
        }
    )
    
    return shade, detailed_shade

# Example 2: Converting between different shade models
def demonstrate_conversions():
    """Example of converting between different shade models"""
    # Create an L1Shade
    l1_shade = L1Shade(
        id="12345",
        user_id="user123",
        name="Programming Languages",
        summary="An overview of various programming languages and their applications.",
        aspect="Technology",
        icon="💻",
        desc_third_view="The user shows interest in multiple programming languages.",
        content_third_view="The user has experience with Python, JavaScript, and Go."
    )
    
    # Convert L1Shade to ShadeInfo
    shade_info = ShadeInfo.from_l1_shade(l1_shade)
    print(f"ShadeInfo name: {shade_info.name}")
    print(f"ShadeInfo content: {shade_info.content}")
    print(f"ShadeInfo aspect: {shade_info.aspect}")
    print(f"ShadeInfo icon: {shade_info.icon}")
    
    # Convert ShadeInfo back to L1Shade
    l1_shade_from_info = shade_info.to_l1_shade(user_id="user123")
    print(f"L1Shade from ShadeInfo - name: {l1_shade_from_info.name}")
    print(f"L1Shade from ShadeInfo - aspect: {l1_shade_from_info.aspect}")
    
    # Create a ShadeMergeInfo from L1Shade
    merge_info = ShadeMergeInfo.from_shade(l1_shade)
    print(f"ShadeMergeInfo name: {merge_info.name}")
    print(f"ShadeMergeInfo aspect: {merge_info.aspect}")
    
    return {
        "original_l1_shade": l1_shade,
        "shade_info": shade_info,
        "l1_shade_from_info": l1_shade_from_info,
        "shade_merge_info": merge_info
    }

# Example 3: Working with timelines
def working_with_timelines():
    """Example of working with timelines in L1Shade"""
    # Create a shade with timelines
    shade = L1Shade(
        id="12345",
        user_id="user123",
        name="Programming Languages",
        summary="An overview of various programming languages and their applications.",
        aspect="Technology",
        icon="💻",
        desc_third_view="The user shows interest in multiple programming languages.",
        content_third_view="The user has experience with Python, JavaScript, and Go.",
        metadata={
            "timelines": [
                {
                    "createTime": "2022-01-15",
                    "description": "Started learning Python",
                    "refMemoryId": "doc123"
                }
            ]
        }
    )
    
    # Access timeline objects using the timelines property
    print("Initial timelines:")
    for timeline in shade.timelines:
        print(f"  {timeline.createTime}: {timeline.desc_third_view} (Ref: {timeline.ref_memory_id})")
    
    # Create a new ShadeTimeline object
    new_timeline = ShadeTimeline(
        refMemoryId="doc456",
        createTime="2022-06-20",
        descThirdView="Started learning JavaScript",
        isNew=True
    )
    
    # Add the timeline to the shade
    shade._timelines.append(new_timeline)
    shade._sync_timelines_to_metadata()
    
    # Use the improve_shade_info method to add more timelines
    shade.improve_shade_info(
        improvedName="",  # Keep the same name
        improvedDesc="An in-depth look at programming languages and frameworks",
        improvedTimelines=[
            {
                "createTime": "2022-09-10",
                "description": "Started learning React",
                "refMemoryId": "doc789"
            }
        ]
    )
    
    # Add second-person view information
    shade.add_second_view(
        domainDesc="You show interest in multiple programming languages.",
        domainContent="You have experience with Python, JavaScript, and Go.",
        domainTimeline=[
            {
                "refMemoryId": "doc123",
                "description": "You started learning Python"
            },
            {
                "refMemoryId": "doc456",
                "description": "You started learning JavaScript"
            }
        ]
    )
    
    # Print all timelines using the ShadeTimeline objects
    print("\nTimeline entries after updates:")
    for timeline in shade.timelines:
        print(f"  {timeline.createTime}: {timeline.desc_third_view} (Ref: {timeline.ref_memory_id})")
        if timeline.desc_second_view:
            print(f"    Second-person view: {timeline.desc_second_view}")
    
    # Print the shade as a string
    print("\nShade as string:")
    print(shade.to_str())
    
    return shade

# Example 4: Interact with the database (example only)
def database_operations(db_session=None):
    """
    Example of database operations with L1Shade.
    
    Note: This is a code example and won't run without a database connection.
    """
    if db_session is None:
        print("This is a code example only. No database operations will be performed.")
        return
    
    # Create a new L1Shade with all LPM Kernel compatible fields
    shade = L1Shade(
        id="12345",
        user_id="user123",
        name="Programming Languages",
        summary="An overview of various programming languages and their applications.",
        confidence=0.9,
        aspect="Technology",
        icon="💻",
        desc_third_view="The user has knowledge of multiple programming languages.",
        content_third_view="The user has explored Python, JavaScript, and Go.",
        desc_second_view="You have knowledge of multiple programming languages.",
        content_second_view="You have explored Python, JavaScript, and Go."
    )
    
    # Create a database model from L1Shade
    db_shade = DBL1Shade(
        id=shade.id,
        user_id=shade.user_id,
        name=shade.name,
        summary=shade.summary,
        confidence=shade.confidence,
        aspect=shade.aspect,
        desc_third_view=shade.desc_third_view,
        content_third_view=shade.content_third_view,
        desc_second_view=shade.desc_second_view,
        content_second_view=shade.content_second_view,
        icon=shade.icon,
        s3_path=f"l1/shades/{shade.user_id}/{shade.id}.json"  # Example path
    )
    
    # Example: Save to database
    # db_session.add(db_shade)
    # db_session.commit()
    
    # Example: Retrieve from database
    # retrieved_shade = db_session.query(DBL1Shade).filter(DBL1Shade.id == shade.id).first()
    
    # Example: Convert DB model to L1Shade
    # l1_shade = L1Shade(
    #     id=retrieved_shade.id,
    #     user_id=retrieved_shade.user_id,
    #     name=retrieved_shade.name,
    #     summary=retrieved_shade.summary,
    #     confidence=retrieved_shade.confidence,
    #     aspect=retrieved_shade.aspect,
    #     icon=retrieved_shade.icon,
    #     desc_third_view=retrieved_shade.desc_third_view,
    #     content_third_view=retrieved_shade.content_third_view,
    #     desc_second_view=retrieved_shade.desc_second_view,
    #     content_second_view=retrieved_shade.content_second_view,
    #     s3_path=retrieved_shade.s3_path
    # )
    
    # Convert to JSON format using LPM Kernel's field naming
    json_representation = shade.to_json()
    print("\nJSON representation with LPM Kernel field naming:")
    print(json.dumps(json_representation, indent=2))
    
    return {
        "original_shade": shade,
        "db_shade": db_shade,
        "json_representation": json_representation
    }

if __name__ == "__main__":
    print("=== Example 1: Creating L1Shade objects ===")
    basic_shade, detailed_shade = create_basic_l1_shade()
    print(f"Basic shade: {basic_shade.name}")
    print(f"Detailed shade: {detailed_shade.name} with confidence {detailed_shade.confidence} and icon {detailed_shade.icon}")
    
    print("\n=== Example 2: Converting between shade models ===")
    conversion_results = demonstrate_conversions()
    
    print("\n=== Example 3: Working with timelines ===")
    timeline_shade = working_with_timelines()
    
    print("\n=== Example 4: Database operations (example only) ===")
    database_operation_results = database_operations()
    
    print("\nAll examples completed successfully!") 