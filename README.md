# Second Me Prototype

Digital Twin Chatbot Prototype that creates a personalized conversational agent based on user documents and data.

## Core Adapters

This project implements core adapters for connecting to various backend services:

1. **BlobStore** (`app/providers/blob_store.py`): An adapter for Wasabi S3 storage that handles document uploading, downloading, and management.

2. **VectorDB** (`app/providers/vector_db.py`): An adapter for Weaviate vector database that handles storing and searching document embeddings with multi-tenancy support.

3. **RelationalDB** (`app/providers/rel_db.py`): An adapter for PostgreSQL that manages structured data like user information, document metadata, training job status, and chat sessions.

## L0 Processing Pipeline

The L0 processing pipeline has been implemented to handle document ingestion and preparation:

1. **Document Processor** (`app/processors/l0/document_processor.py`): Orchestrates the entire L0 pipeline for processing documents.

2. **Content Extractor** (`app/processors/l0/content_extractor.py`): Extracts text content from various file formats including PDF, DOCX, and plain text.

3. **Document Analyzer** (`app/processors/l0/document_analyzer.py`): Generates insights, summaries, and titles from document content using AI.

4. **Chunker** (`app/processors/l0/chunker.py`): Segments documents into semantic chunks for effective retrieval.

5. **Embedding Generator** (`app/processors/l0/embedding_generator.py`): Creates vector embeddings for document chunks using OpenAI's embedding models.

The pipeline processes documents through these steps:
- Extracts content from uploaded files
- Analyzes documents to generate insights and summaries
- Stores the original document and analysis in both Wasabi and PostgreSQL
- Chunks the content into semantically meaningful sections
- Generates embeddings for each chunk
- Stores chunks in Wasabi and their embeddings in Weaviate vector database

## Environment Setup

Copy the `.env.example` file to `.env` and fill in the required values:

```bash
cp .env.example .env
# Edit the .env file with your service credentials
```

Required environment variables:

- **Wasabi S3**: `WASABI_ACCESS_KEY`, `WASABI_SECRET_KEY`, `WASABI_BUCKET`, `WASABI_REGION`, `WASABI_ENDPOINT`
- **Weaviate Cloud**: `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `EMBEDDING_MODEL`
- **PostgreSQL**: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- **OpenAI API**: `OPENAI_API_KEY`

## Installation

1. Create a virtual environment:

```bash
python --version # this should be python 3.12!
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Testing

### Testing Core Adapters

Run the adapter tests to verify connections to backend services:

```bash
python tests/test_adapters.py
```

This will test:
- Wasabi S3 connections and operations
- Weaviate vector database connections and operations
- PostgreSQL database connections and operations

### Testing L0 Pipeline

To test the L0 processing pipeline with a sample document:

```bash
python scripts/test_l0_pipeline.py [path_to_document]
```

If no document path is specified, it will use a default sample document.

## Project Structure

```
second-me-prototype/
├── app/                       # Application code
│   ├── core/                  # Core settings and dependencies
│   │   ├── config.py          # Environment and application settings
│   │   └── dependencies.py    # Dependency injection
│   ├── providers/             # Service adapters
│   │   ├── blob_store.py      # Wasabi S3 adapter
│   │   ├── vector_db.py       # Weaviate adapter
│   │   └── rel_db.py          # PostgreSQL adapter
│   ├── processors/            # Processing pipelines
│   │   ├── l0/                # L0 processing (document ingestion)
│   │   │   ├── chunker.py     # Document chunking
│   │   │   ├── content_extractor.py # Content extraction
│   │   │   ├── document_analyzer.py # Document analysis
│   │   │   ├── document_processor.py # Pipeline orchestration
│   │   │   ├── embedding_generator.py # Embedding generation
│   │   │   ├── models.py      # Data models
│   │   │   └── utils.py       # Utility functions
│   │   └── __init__.py
│   └── api/                   # API routes (to be implemented)
├── scripts/                   # Utility scripts
│   └── test_l0_pipeline.py    # Script for testing L0 pipeline
├── tests/                     # Test scripts
│   ├── test_adapters.py       # Adapter tests
│   └── processors/            # Processor tests
│       └── l0/                # L0 pipeline tests
├── design_docs/               # Design documentation
│   ├── v0_architecture.md     # System architecture
│   └── v0_L0_instructions.md  # L0 implementation instructions
└── .env.example               # Example environment variables
``` 