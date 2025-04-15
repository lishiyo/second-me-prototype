# L0 Flow Summary

File ingestion:
- grab the files from `data` (future: User uploads files)
- Raw file stored in Wasabi
- Basic metadata stored in PostgreSQL

Content Extraction:
- Text extracted from file based on MIME type
- Raw content stored in Wasabi

Document Analysis:
- LLM generates insights and title
- Results stored in Wasabi

Chunking:
- Document split into chunks based on token count
- Chunks stored in Wasabi

Embedding Generation:
- Embeddings generated for each chunk (use our own OpenAI embeddings to generate the embedding for the text chunk)
- Document-level embedding generated
- Embeddings stored in Weaviate with metadata links, pointing to Wasabi for the chunk text

Vector Search:
- Query converted to embedding
- Similar chunks retrieved from Weaviate
- Full context retrieved from Wasabi

# L0 Processing Pipeline Implementation Plan

## Overview

The L0 processing pipeline is responsible for the initial processing of user documents:
1. Ingesting raw documents
2. Extracting text content
3. Analyzing documents to generate insights and titles
4. Chunking documents into smaller pieces
5. Generating embeddings for chunks
6. Storing all data in appropriate storage systems

## Architecture Alignment

Our implementation will use the following components:
- **Wasabi S3**: For raw document storage, chunk storage, and metadata
- **Weaviate**: For storing embeddings and pointers to chunks in Wasabi (no content storage)
- **PostgreSQL**: For document metadata and processing status tracking
- **OpenAI API**: For generating embeddings and insights

## Directory Structure

```
app/
├── processors/
│   ├── __init__.py
│   ├── l0/
│   │   ├── __init__.py
│   │   ├── document_processor.py      # Main processor class
│   │   ├── content_extractor.py       # Content extraction utilities
│   │   ├── document_analyzer.py       # Insight generation using OpenAI
│   │   ├── chunker.py                 # Text chunking implementation
│   │   ├── embedding_generator.py     # Creates document and chunk embeddings
│   │   └── models.py                  # Input/output data models
```

## Implementation Plan

### 1. Document Processor Class (`document_processor.py`)

The main orchestrator that:
- Takes a document ID or path as input
- Calls each processing stage in sequence
- Handles error recovery and retries
- Updates processing status in PostgreSQL

```python
class DocumentProcessor:
    def __init__(self, blob_store, vector_db, rel_db):
        self.blob_store = blob_store
        self.vector_db = vector_db
        self.rel_db = rel_db
        self.content_extractor = ContentExtractor()
        self.document_analyzer = DocumentAnalyzer()
        self.chunker = Chunker()
        self.embedding_generator = EmbeddingGenerator()
    
    def process_document(self, document_id, user_id="1"):
        """Process a document through the entire L0 pipeline."""
        # Get document metadata from PostgreSQL
        # Extract content
        # Analyze document
        # Chunk document
        # Generate embeddings
        # Store in Weaviate
```

### 2. Content Extractor (`content_extractor.py`)

Handles different file types:
- Plain text
- PDF
- MS Office documents
- HTML/Markdown

```python
class ContentExtractor:
    def extract_content(self, file_content, mime_type):
        """Extract text content based on file type."""
        # Use appropriate extraction method based on mime_type
```

### 3. Document Analyzer (`document_analyzer.py`)

Uses OpenAI to generate document insights:
- Title generation
- Summary generation
- Keyword extraction
- Topic identification

```python
class DocumentAnalyzer:
    def generate_insights(self, text_content, filename):
        """Generate title, summary, and keywords using LLM."""
        # Call OpenAI API to generate insights
```

### 4. Chunker (`chunker.py`)

Implements the `TokenTextSplitter` and `TokenParagraphSplitter` from lpm_kernel:
- Semantic chunking based on token count and paragraph structure
- Smart boundary detection to avoid cutting in the middle of ideas
- Handling of URLs and special content

```python
class Chunker:
    def chunk_document(self, content, chunk_size=512, chunk_overlap=50):
        """Split document into semantic chunks."""
        # Use algorithm similar to TokenParagraphSplitter
```

### 5. Embedding Generator (`embedding_generator.py`)

Generates embeddings for chunks and documents:
- Uses OpenAI embeddings API
- Handles batching for efficiency
- Ensures proper dimension consistency

```python
class EmbeddingGenerator:
    def generate_embeddings(self, text_chunks):
        """Generate embeddings for text chunks using OpenAI API."""
        # Call OpenAI API to generate embeddings
```

## Storage Model

### Wasabi S3 Storage Structure

```
tenant/1/                           # For MVP, only user_id = 1
  ├── raw/                          # Original documents
  │   ├── doc123.pdf
  │   └── ...
  ├── chunks/                       # Processed document chunks
  │   ├── doc123/
  │   │   ├── chunk_0.txt
  │   │   ├── chunk_1.txt
  │   │   └── ...
  │   └── ...
  └── metadata/                     # Document analysis data
      ├── doc123/
      │   ├── insight.json         # title, insight
      │   └── summary.json         # title, summary, keywords
      └── ...
```

### Weaviate Schema (Multi-Tenant Design)

```
TenantChunk {
  document_id: String            # Source document ID
  s3_path: String                # Direct path to chunk file in Wasabi
  chunk_index: Integer           # Index of chunk in document
  metadata: Object {             # Additional metadata
    filename: String
    content_type: String
    timestamp: DateTime
  }
}
```

### PostgreSQL Document Model

```
Document {
  id: UUID                       # Unique document ID
  user_id: UUID                  # User ID (always "1" for MVP)
  filename: String               # Original filename
  content_type: String           # MIME type
  s3_path: String                # Path to document in Wasabi
  uploaded_at: DateTime          # Upload timestamp
  processed: Boolean             # Processing status
  chunk_count: Integer           # Number of chunks generated
}
```

## Processing Flow

1. **Document Ingestion**:
   - File is uploaded to the `data` directory
   - Metadata is stored in PostgreSQL
   - Raw file is uploaded to Wasabi (`tenant/1/raw/`)
   - Processing status is updated in PostgreSQL

2. **Content Extraction**:
   - Raw file is retrieved from Wasabi
   - Text is extracted based on MIME type
   - Full content is stored back in Wasabi (if needed)

3. **Document Analysis**:
   - OpenAI API is called to generate insights
   - Title, summary, and keywords are extracted
   - Results are stored in Wasabi (`tenant/1/metadata/`)
   - Document record is updated in PostgreSQL

4. **Chunking**:
   - Document content is chunked using TokenParagraphSplitter
   - Each chunk is stored as a separate file in Wasabi (`tenant/1/chunks/`)
   - Chunk count is updated in PostgreSQL

5. **Embedding Generation**:
   - OpenAI API is called to generate embeddings for each chunk
   - Document-level embedding is generated
   - Embeddings + metadata + S3 pointers are stored in Weaviate
   - No actual chunk text is stored in Weaviate

6. **Vector Search**:
   - Search query is converted to an embedding
   - Weaviate returns relevant chunk metadata and S3 paths
   - Chunk content is retrieved from Wasabi as needed

## Error Handling

- Implement retry mechanism for API calls (up to 3 attempts)
- Store processing status in PostgreSQL for tracking
- Log detailed error information for debugging
- Enable resuming processing from the last successful step

## Implementation Sequence

1. Create the directory structure and basic class skeletons
2. Implement the Chunker first, adapting code from lpm_kernel
3. Add the ContentExtractor for basic file types
4. Implement the EmbeddingGenerator using OpenAI API
5. Build DocumentAnalyzer for generating insights
6. Create the main DocumentProcessor to orchestrate the process
7. Add comprehensive tests for each component
8. Integrate with the FastAPI backend

## Performance Considerations

- Batch embedding generation for efficiency
- Use async processing for I/O-bound operations
- Implement proper connection pooling for database operations
- Monitor and optimize S3 operations to minimize costs
- Use streaming for large file processing
