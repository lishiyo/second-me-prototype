"""
L0 Processing Pipeline for document chunking, embedding, and analysis.
This module handles the initial processing of documents, including:
- Content extraction from various file formats
- Document analysis to generate insights
- Semantic chunking of document text
- Embedding generation for chunks
- Storage of raw documents, chunks, and embeddings
"""

from app.processors.l0.models import (
    ProcessingStatus,
    FileInfo,
    ChunkInfo,
    DocumentInsights,
    ProcessingResult
)

from app.processors.l0.content_extractor import ContentExtractor
from app.processors.l0.chunker import Chunker
from app.processors.l0.document_analyzer import DocumentAnalyzer
from app.processors.l0.embedding_generator import EmbeddingGenerator
from app.processors.l0.document_processor import DocumentProcessor

__all__ = [
    # Data models
    "ProcessingStatus",
    "FileInfo",
    "ChunkInfo",
    "DocumentInsights",
    "ProcessingResult",
    
    # Components
    "ContentExtractor",
    "Chunker",
    "DocumentAnalyzer",
    "EmbeddingGenerator",
    "DocumentProcessor"
]
