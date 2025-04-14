"""
Simple example showing how to use our document analyzer with the new prompts.
"""

import os
import json
import sys
from typing import Dict, Any

# Try to make imports work from different contexts
try:
    # When running from project root
    from app.processors.l0.document_analyzer import DocumentInsightGenerator, DocumentSummaryGenerator
    from app.processors.l0.document_insight_processor import DocumentInsightProcessor
    from app.processors.l0.models import FileInfo, BioInfo, InsighterInput
except ModuleNotFoundError:
    # Add project root to the path if running from examples directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    try:
        # Try again with project root in path
        from app.processors.l0.document_analyzer import DocumentInsightGenerator, DocumentSummaryGenerator
        from app.processors.l0.document_insight_processor import DocumentInsightProcessor
        from app.processors.l0.models import FileInfo, BioInfo, InsighterInput
    except ModuleNotFoundError:
        # Last resort - try relative imports
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
        from document_analyzer import DocumentInsightGenerator, DocumentSummaryGenerator
        from document_insight_processor import DocumentInsightProcessor
        from models import FileInfo, BioInfo, InsighterInput

def main():
    """Run a simple example of the document analyzer."""
    # Ensure OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        return
        
    # Sample document content
    sample_document = """
    # Machine Learning: An Overview
    
    Machine learning is a subset of artificial intelligence that provides systems with the ability to 
    automatically learn and improve from experience without being explicitly programmed.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    
    Supervised learning algorithms build a mathematical model of a set of data that contains both 
    the inputs and the desired outputs. Examples include classification and regression.
    
    ### Unsupervised Learning
    
    Unsupervised learning algorithms take a set of data that contains only inputs, and find structure
    in the data, like grouping or clustering data points. Examples include clustering and association.
    
    ### Reinforcement Learning
    
    Reinforcement learning is an area of machine learning concerned with how software agents ought 
    to take actions in an environment in order to maximize some notion of cumulative reward.
    
    ## Applications
    
    Machine learning has been used in numerous fields including:
    
    - Computer vision
    - Natural language processing
    - Recommendation systems
    - Healthcare diagnostics
    - Financial market analysis
    """
    
    # Sample biographical information
    sample_bio = BioInfo(
        about_me="I'm a software engineer with an interest in AI and machine learning.",
        global_bio="Works in the tech industry. Enjoys learning about new technologies and programming languages.",
        status_bio="Recently started learning about deep learning frameworks like TensorFlow and PyTorch."
    )
    
    # Create file info
    file_info = FileInfo(
        document_id="doc123",
        filename="machine_learning_overview.md",
        content_type="text/markdown",
        s3_path="documents/machine_learning_overview.md",
        content=sample_document
    )
    
    # Method 1: Direct use of DocumentInsightGenerator
    print("\n=== Method 1: Using DocumentInsightGenerator directly ===")
    insight_generator = DocumentInsightGenerator()
    insight = insight_generator.generate_insight(sample_document, "machine_learning_overview.md")
    print(f"Title: {insight.title}")
    print(f"Insight: {insight.insight[:100]}...\n")
    
    # Method 2: Using DocumentInsightProcessor (without bio)
    print("\n=== Method 2: Using DocumentInsightProcessor (without bio) ===")
    processor = DocumentInsightProcessor()
    result = processor.process_document(sample_document, "machine_learning_overview.md")
    print(f"Insight Title: {result['insight'].title}")
    print(f"Insight: {result['insight'].insight[:100]}...")
    print(f"Summary: {result['summary'].summary[:100]}...")
    print(f"Keywords: {', '.join(result['summary'].keywords)}\n")
    
    # Method 3: Using the processor with bio information
    print("\n=== Method 3: Using DocumentInsightProcessor with bio info ===")
    result = processor.process_with_bio(file_info, sample_bio)
    print(f"Bio-Enhanced Insight Title: {result['insight'].title}")
    print(f"Bio-Enhanced Insight: {result['insight'].insight[:100]}...")
    print(f"Bio-Enhanced Summary: {result['summary'].summary[:100]}...")
    print(f"Bio-Enhanced Keywords: {', '.join(result['summary'].keywords)}\n")
    
    # Method 4: Using InsighterInput
    print("\n=== Method 4: Using InsighterInput ===")
    input_data = InsighterInput(file_info=file_info, bio_info=sample_bio)
    result = processor.process_from_insighter_input(input_data)
    print(f"From InsighterInput - Title: {result['insight'].title}")
    print(f"From InsighterInput - Insight: {result['insight'].insight[:100]}...")
    
    # Method 5: Using DocumentSummaryGenerator's generate_title_and_summary method
    print("\n=== Method 5: Using DocumentSummaryGenerator directly ===")
    summary_generator = DocumentSummaryGenerator()
    result = summary_generator.generate_title_and_summary(sample_document, "machine_learning_overview.md")
    print(f"Direct Title: {result['title']}")
    print(f"Direct Summary: {result['summary'][:100]}...")
    print(f"Direct Keywords: {', '.join(result['keywords'])}")

if __name__ == "__main__":
    main() 