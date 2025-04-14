# L0 Processing Pipeline Implementation Instructions

This document provides detailed instructions for implementing the L0 processing pipeline, which handles the initial processing of documents for the digital twin system. The implementation follows the design outlined in `v0_L0_flow.md` and adapts the example code from `lpm_kernel` to our stack.

## Implementation Subtasks Overview

The L0 pipeline implementation is broken down into the following subtasks:

1. Setup Directory Structure
2. Implement Data Models and Utilities 
3. Implement Core Processing Components
   - 3a. Chunker Implementation
   - 3b. Content Extractor Implementation
   - 3c. Embedding Generator Implementation
   - 3d. Document Analyzer Implementation
4. Implement Document Processor (Orchestration)
5. Integrate with Storage Adapters
6. Implement Testing
7. Expose through FastAPI

## Dependencies Map

```
Subtask 1 (Setup) → All other subtasks
Subtask 2 (Models) → Subtasks 3a-d, 4, 5, 6
Subtasks 3a-d (Components) → Subtask 4
Subtask 4 (Processor) → Subtask 5
Subtask 5 (Integration) → Subtask 6
Subtasks 1-6 → Subtask 7
```

## Parallel Development Opportunities

- Subtasks 3a-d can be developed in parallel after completing Subtasks 1 and 2
- Unit tests (part of Subtask 6) can be developed alongside each component
- API routes (Subtask 7) can be started after the basic processor structure is defined

## Detailed Instructions by Subtask

### Subtask 1: Setup Directory Structure

**Steps:**
1. Create the app/processors directory if it doesn't exist
   ```bash
   mkdir -p app/processors
   ```

2. Create app/processors/__init__.py
   ```bash
   touch app/processors/__init__.py
   ```

3. Create app/processors/l0/ directory
   ```bash
   mkdir -p app/processors/l0
   ```

4. Create app/processors/l0/__init__.py
   ```bash
   touch app/processors/l0/__init__.py
   ```

5. Create placeholder files for each module:
   ```bash
   touch app/processors/l0/models.py
   touch app/processors/l0/chunker.py
   touch app/processors/l0/content_extractor.py
   touch app/processors/l0/document_analyzer.py
   touch app/processors/l0/embedding_generator.py
   touch app/processors/l0/document_processor.py
   ```

**Dependencies:** None

**Expected Output:** Directory structure ready for implementing the L0 pipeline components

### Subtask 2: Implement Data Models and Utilities

**Steps:**
1. Study lpm_kernel/L0/models.py to understand the existing data models
2. In app/processors/l0/models.py, create the following classes:
   - FileInfo class to represent document metadata
   - ProcessingResult class to track processing status
   - ChunkInfo class to represent document chunks
   - Create enums for processing status (Queued, Processing, Completed, Failed)

3. Implement utility functions in app/processors/l0/utils.py:
   - Retry mechanism for API calls
   - Error handling helpers
   - Logging configuration

**Dependencies:** Subtask 1

**Expected Output:** Data models and utilities that can be used by other components

### Subtask 3a: Implement Chunker

**Steps:**
1. Study TokenParagraphSplitter and TokenTextSplitter in lpm_kernel/utils.py
2. In app/processors/l0/chunker.py, create a Chunker class with the following methods:
   - __init__ method that initializes a tokenizer (using tiktoken)
   - chunk_document method that splits text into semantic chunks
   - Methods to handle special cases (URLs, code blocks)
   - Helper methods for text preprocessing and postprocessing

3. Add configuration options for chunk size and overlap

4. Implement unit tests for the Chunker class

**Dependencies:** Subtask 1, Subtask 2

**Expected Output:** A Chunker class that can split documents into semantic chunks

### Subtask 3b: Implement Content Extractor

**Steps:**
1. In app/processors/l0/content_extractor.py, create a ContentExtractor class with the following methods:
   - extract_content method that takes file content and mime type as input
   - Specific extraction methods for different file types:
     - extract_text_from_plain_text
     - extract_text_from_pdf (using PyPDF2 or similar)
     - extract_text_from_docx (using python-docx)
     - extract_text_from_html (using BeautifulSoup)

2. Add fallback mechanisms for unknown formats
3. Handle encoding issues and text cleanup
4. Implement unit tests for the ContentExtractor class

**Dependencies:** Subtask 1, Subtask 2

**Expected Output:** A ContentExtractor class that can extract text from different file types

### Subtask 3c: Implement Embedding Generator

**Steps:**
1. In app/processors/l0/embedding_generator.py, create an EmbeddingGenerator class with the following methods:
   - __init__ method that configures the OpenAI client
   - generate_embedding method for a single text
   - generate_batch_embeddings method for multiple chunks
   - Methods for handling API rate limits and retries

2. Add configuration for the embedding model (text-embedding-3-small)
3. Implement unit tests for the EmbeddingGenerator class

**Dependencies:** Subtask 1, Subtask 2

**Expected Output:** An EmbeddingGenerator class that can generate embeddings for text chunks

### Subtask 3d: Implement Document Analyzer

**Steps:**
1. Study the document analysis code in lpm_kernel/L0/l0_generator.py
2. In app/processors/l0/document_analyzer.py, create a DocumentAnalyzer class with the following methods:
   - __init__ method that configures the OpenAI client
   - generate_insights method that analyzes document content
   - Methods for extracting title, summary, and keywords
   - Helper methods for prompt formatting

3. Implement unit tests for the DocumentAnalyzer class

**Dependencies:** Subtask 1, Subtask 2

**Expected Output:** A DocumentAnalyzer class that can generate insights from document content

### Subtask 4: Implement Document Processor

**Steps:**
1. In app/processors/l0/document_processor.py, create a DocumentProcessor class with the following methods:
   - __init__ method that initializes storage adapters and processing components
   - process_document method that orchestrates the processing pipeline:
     - Get document metadata from PostgreSQL
     - Retrieve document from Wasabi
     - Extract content using ContentExtractor
     - Generate insights using DocumentAnalyzer
     - Chunk document using Chunker
     - Generate embeddings using EmbeddingGenerator
     - Store results in appropriate storage systems
   - Methods for error handling and progress tracking

2. Implement unit tests for the DocumentProcessor class

**Dependencies:** Subtask 3a, Subtask 3b, Subtask 3c, Subtask 3d

**Expected Output:** A DocumentProcessor class that orchestrates the entire L0 pipeline

### Subtask 5: Integrate with Storage Adapters

**Steps:**
1. Update DocumentProcessor to use our storage adapters:
   - Use BlobStore for Wasabi operations:
     - Store raw documents in tenant/1/raw/
     - Store chunks in tenant/1/chunks/{doc_id}/
     - Store metadata in tenant/1/metadata/{doc_id}/
   - Use VectorDB for Weaviate operations:
     - Store chunk embeddings with metadata and s3_path pointers
   - Use RelationalDB for PostgreSQL operations:
     - Update document processing status
     - Track document metadata

2. Implement error handling for storage operations
3. Add transaction support where appropriate
4. Implement integration tests for storage operations

**Dependencies:** Subtask 4

**Expected Output:** DocumentProcessor that integrates with all storage systems

### Subtask 6: Implement Testing

**Steps:**
1. Create test directory structure:
   ```bash
   mkdir -p tests/processors/l0
   touch tests/processors/l0/__init__.py
   ```

2. Create test fixtures:
   - Sample documents of different types
   - Mock responses for API calls

3. Implement unit tests for each component:
   - tests/processors/l0/test_chunker.py
   - tests/processors/l0/test_content_extractor.py
   - tests/processors/l0/test_embedding_generator.py
   - tests/processors/l0/test_document_analyzer.py
   - tests/processors/l0/test_document_processor.py

4. Implement integration tests:
   - tests/processors/l0/test_l0_integration.py

**Dependencies:** Subtasks 1-5

**Expected Output:** Comprehensive test suite for the L0 pipeline

### Subtask 7: Expose through FastAPI

**NOTE**: This has been skipped for now, we will focus on L1.

**Steps:**
1. Create or update API routes in app/api/ingest.py:
   - POST endpoint for document ingestion
   - GET endpoint for checking processing status
   - GET endpoint for retrieving document metadata from our relational db
   - GET endpoint for searching with a query and retrieving the relevant chunks
   - GET endpoint for searching with a query and retrieving an answer from OpenAI

2. Implement background task processing:
   - Use Redis queue for document processing tasks
   - Create worker that runs DocumentProcessor

3. Add API documentation with OpenAPI
4. Implement proper error responses and status codes
5. Test API endpoints

**Dependencies:** Subtasks 1-6

**Expected Output:** API endpoints for the L0 pipeline

## Implementation Notes and Considerations

### Potential Challenges

1. **Semantic Chunking Strategy**: Finding the right balance between semantic coherence and chunk size will be crucial. This may require experimentation with different chunking approaches.

2. **Embedding Model Performance**: The embedding generation can become a performance bottleneck, especially for large documents. Consider implementing batching and parallel processing.

3. **Storage Efficiency**: Storing both raw documents and chunks can consume significant storage space. Consider compression or selective storage strategies.

4. **Error Handling**: Robust error handling throughout the pipeline is crucial to prevent data loss or corruption. Implement comprehensive retry mechanisms and validation checks.

5. **Integration with Future Components**: Ensure the L0 pipeline outputs data in a format that can be consumed by future pipeline stages (L1, L2).

### Optimization Opportunities

1. **Batch Processing**: Implement batch processing for embedding generation and Weaviate operations to improve throughput.

2. **Caching**: Consider caching embeddings for frequently accessed chunks or documents.

3. **Async Processing**: Use asynchronous processing for I/O-bound operations (API calls, database operations).

4. **Streaming**: Implement streaming for large file processing to reduce memory usage.

5. **Monitoring**: Add performance monitoring to identify bottlenecks in the pipeline.

## Implementation Sequence

The recommended implementation sequence is:

1. Subtask 1: Setup Directory Structure
2. Subtask 2: Implement Data Models and Utilities
3. Subtasks 3a, 3b, 3c, 3d (can be implemented in parallel)
4. Subtask 4: Implement Document Processor
5. Subtask 5: Integrate with Storage Adapters
6. Subtask 6: Implement Testing
7. Subtask 7: Expose through FastAPI

This sequence allows for incremental development and testing of the L0 pipeline. 