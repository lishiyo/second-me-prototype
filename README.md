# Second Me Prototype

Digital Twin Chatbot Prototype that creates a personalized conversational agent based on user documents and data.

## Core Adapters

This project implements core adapters for connecting to various backend services:

1. **BlobStore** (`app/providers/blob_store.py`): An adapter for Wasabi S3 storage that handles document uploading, downloading, and management.

2. **VectorDB** (`app/providers/vector_db.py`): An adapter for Weaviate vector database that handles storing and searching document embeddings with multi-tenancy support.

3. **RelationalDB** (`app/providers/rel_db.py`): An adapter for PostgreSQL that manages structured data like user information, document metadata, training job status, and chat sessions.

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

## Testing Adapters

Run the adapter tests to verify connections to backend services:

```bash
python tests/test_adapters.py
```

This will test:
- Wasabi S3 connections and operations
- Weaviate vector database connections and operations
- PostgreSQL database connections and operations

## Project Structure

```
second-me-prototype/
├── app/                       # FastAPI service
│   ├── core/                  # Core settings and dependencies
│   │   ├── config.py          # Environment and application settings
│   │   └── dependencies.py    # Dependency injection
│   ├── providers/             # Service adapters
│   │   ├── blob_store.py      # Wasabi S3 adapter
│   │   ├── vector_db.py       # Weaviate adapter
│   │   └── rel_db.py          # PostgreSQL adapter
│   └── api/                   # API routes (to be implemented)
├── tests/                     # Test scripts
│   └── test_adapters.py       # Adapter tests
└── .env.example               # Example environment variables
``` 