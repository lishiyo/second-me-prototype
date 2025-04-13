# Digital Twin Chatbot Architecture

## System Architecture Overview

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────────┐
│                │     │                 │     │                   │
│  User/Client   │────▶│  FastAPI Server │────▶│  Redis Queue      │
│                │     │                 │     │                   │
└────────────────┘     └─────────────────┘     └───────────────────┘
                              │  ▲                      │
                              │  │                      │
                              ▼  │                      ▼
┌────────────────┐     ┌─────────────────┐     ┌───────────────────┐
│                │     │                 │     │                   │
│  PostgreSQL    │◀───▶│  Modal Function │◀───▶│  Worker Processes │
│                │     │                 │     │                   │
└────────────────┘     └─────────────────┘     └───────────────────┘
        ▲                      │  ▲                      │
        │                      │  │                      │
        │                      ▼  │                      ▼
┌────────────────┐     ┌─────────────────┐     ┌───────────────────┐
│                │     │                 │     │                   │
│  Weaviate      │◀───▶│  Wasabi S3      │◀───▶│  OpenAI API       │
│                │     │                 │     │                   │
└────────────────┘     └─────────────────┘     └───────────────────┘
```

## Component Details

### Core Components

1. **FastAPI Server**
   - Entry point for all client requests
   - Routes for document uploading, chat interactions
   - Manages authentication (future enhancement, ignore this for now)
   - Delegates heavy processing to workers via Redis

2. **Redis Queue**
   - Manages background job queues
   - Separate queues for ingestion and training
   - Enables asynchronous processing

3. **Worker Processes**
   - Document Ingestion Worker (processes new documents)
   - Training Worker (schedules and monitors model training)
   - Handles retries for failed jobs (up to 3 attempts)

4. **PostgreSQL Database**
   - Stores user information
   - Tracks document metadata
   - Records training job status
   - Tracks minimal chat session metadata (paths to files in Wasabi)
   - Uses row-level security (future enhancement)

5. **Weaviate Vector Database**
   - Stores document chunk embeddings and metadata only (not the actual chunk text)
   - Maintains pointers (s3_path) to retrieve chunk text from Wasabi
   - Enables semantic search over the embeddings
   - Multi-tenant architecture
   - Supports efficient retrieval for RAG

6. **Wasabi S3 Storage**
   - Stores raw documents and pre-processed chunks separately
   - Stores fine-tuned model weights
   - Organizes data by user ID
   - Minimizes egress costs by co-locating with Modal

7. **Modal Functions**
   - Serverless execution environment
   - Training function (GPU-accelerated)
   - Inference function (GPU-accelerated)
   - Shared volumes for model caching

8. **OpenAI API**
   - Provides embeddings for document chunks
   - Used for chat completions during training data generation

## Data Flows

### Document Ingestion Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │     │         │
│ FastAPI │────▶│ Redis   │────▶│ Ingest  │────▶│ L0      │────▶│ L1      │
│ Server  │     │ Queue   │     │ Worker  │     │ Process │     │ Process │
│         │     │         │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                     │               │
                                                     ▼               ▼
                                                ┌─────────┐     ┌─────────┐
                                                │         │     │         │
                                                │ Wasabi  │     │ Weaviate│
                                                │ S3      │     │         │
                                                │         │     │         │
                                                └─────────┘     └─────────┘
```

1. User uploads document to FastAPI server
2. Server validates document and enqueues processing job
3. Ingest Worker picks up job and initiates processing
4. L0 Process:
   - Uploads original document to Wasabi raw folder
   - Chunks document by text semantics
   - Stores each chunk as a separate file in Wasabi chunks folder
   - Embeds chunks using OpenAI embeddings
   - Stores embeddings and pointers (s3_path) in Weaviate, but not the chunk text itself
5. L1 Process:
   - Organizes chunks into hierarchical memory structure
   - Creates topic clusters
   - Stores memory structure in Wasabi

### Training Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │
│ FastAPI │────▶│ Redis   │────▶│ Training│────▶│ Modal   │
│ Server  │     │ Queue   │     │ Worker  │     │ Training│
│         │     │         │     │         │     │ Function│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                     │
                                                     ▼
┌─────────┐                                     ┌─────────┐
│         │                                     │         │
│ Postgres│◀────────────────────────────────────│ L2      │
│ DB      │                                     │ Process │
│         │                                     │         │
└─────────┘                                     └─────────┘
                                                     │
                                                     ▼
                                                ┌─────────┐
                                                │         │
                                                │ Wasabi  │
                                                │ S3      │
                                                │         │
                                                └─────────┘
```

1. Training can be triggered manually via API
2. Training Worker prepares and schedules Modal training job
3. Modal Training Function:
   - Loads hierarchical memory structure from Wasabi
   - L2 Process:
     - Generates synthetic training data
     - Downloads base model
     - Fine-tunes model using LoRA
     - Converts to GGUF format
   - Uploads model weights to Wasabi
4. Updates PostgreSQL with model status
5. Notifies inference service of model update

### Inference Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│         │     │         │     │         │     │         │
│ User    │────▶│ FastAPI │────▶│ Modal   │────▶│ Weaviate│
│ Client  │     │ Server  │     │ Inference│    │         │
│         │     │         │     │ Function │    │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                   │    ▲
                                   │    │
                                   ▼    │
                              ┌─────────┐
                              │         │
                              │ Postgres│
                              │ DB      │
                              │         │
                              └─────────┘
                                   │
                                   ▼
                              ┌─────────┐
                              │         │
                              │ Wasabi  │────▶ Chunk Retrieval
                              │ S3      │
                              │         │
                              └─────────┘
```

1. User sends chat message to FastAPI server
2. Server forwards message to Modal Inference Function
3. Inference Function:
   - Loads fine-tuned model for user (caching)
   - Performs semantic search in Weaviate to find relevant chunk embeddings
   - Retrieves the pre-processed chunks directly from Wasabi using the s3_path pointers
   - Generates response using model + retrieved context
   - Stores chat session as structured files in Wasabi
4. Response is returned to user

## Storage Structure

### Wasabi S3 Layout

```
tenant/1/                              # User ID 1 (only user for MVP)
  ├── raw/                             # Original documents
  │   ├── doc1.pdf
  │   ├── doc2.txt
  │   └── ...
  ├── chunks/                          # Processed document chunks as individual files
  │   ├── doc1/
  │   │   ├── chunk_0.txt
  │   │   ├── chunk_1.txt
  │   │   └── ...
  │   └── ...
  ├── chats/                           # Chat session files 
  │   ├── session_123/                 # Individual chat sessions
  │   │   ├── metadata.json            # Session metadata (title, summary, creation time)
  │   │   ├── messages.json            # Full chat transcript with user/assistant messages
  │   │   └── processed/               # Processed chat data for training
  │   │       ├── chat_0.txt           # Formatted chat for training
  │   │       └── ...
  │   └── ...
  ├── training_data/                   # Expanded training data structure
  │   ├── L1/                          # L1 processing outputs
  │   │   ├── processed_data/          # Processed data files
  │   │   │   ├── subjective/          # Subjective training data
  │   │   │   │   ├── chat_0.txt
  │   │   │   │   ├── doc_0.txt
  │   │   │   │   └── ...
  │   │   │   └── objective/           # Objective training data
  │   │   │       ├── chat_0.txt
  │   │   │       └── ...
  │   │   └── graphrag_indexing/       # GraphRAG indexing outputs
  │   │       ├── subjective/
  │   │       │   ├── entities.parquet
  │   │       │   └── relations.parquet
  │   │       └── objective/
  │   │           └── ...
  │   └── L2/                          # L2 processing outputs
  │       ├── data_pipeline/           # Pipeline outputs
  │       │   ├── raw_data/            # Raw training data
  │       │   │   ├── preference.json
  │       │   │   ├── diversity.json
  │       │   │   ├── selfqa.json
  │       │   │   └── merged.json
  │       │   └── mapping/             # Entity and ID mappings
  │       │       ├── id_entity_mapping_subjective_v2.json
  │       │       └── ...
  │       └── training_inputs/         # Final training inputs
  │           ├── dataset.json         # Merged training dataset
  │           └── ...
  ├── metadata/                        # L1 hierarchical structure
  │   ├── topics.json
  │   ├── entities.json
  │   └── ...
  └── lora/                            # Training artifacts
      ├── run_123/                     # Specific training run
      │   ├── adapter.safetensors
      │   ├── merged.gguf
      │   └── training_config.json     # Training configuration used
      ├── run_456/
      │   └── ...
      └── latest/                      # Symlinks to latest
          ├── adapter.safetensors
          └── merged.gguf
```

### Weaviate Schema

We are using the Weaviate v4 python client with multi-tenancy.

```
TenantChunk {
  document_id: String            # Source document ID
  s3_path: String                # Direct path to chunk file in Wasabi
  metadata: Object {             # Additional metadata
    filename: String
    content_type: String
    timestamp: DateTime
    chunk_index: Integer         # Index of chunk in document
  }
}
```

### PostgreSQL Schema

```sql
-- Users table (just user 1 for MVP)
CREATE TABLE users (
  id STRING PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Documents metadata
CREATE TABLE documents (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  filename TEXT NOT NULL,
  content_type TEXT NOT NULL,
  s3_path TEXT NOT NULL,
  uploaded_at TIMESTAMP NOT NULL DEFAULT NOW(),
  processed BOOLEAN DEFAULT FALSE,
  chunk_count INTEGER DEFAULT 0
);

-- Training jobs
CREATE TABLE training_jobs (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  status TEXT NOT NULL,  -- 'queued', 'processing', 'completed', 'failed'
  attempt INT DEFAULT 1,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  lora_path TEXT,
  error TEXT
);

-- Chat session metadata (minimal tracking)
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  title TEXT,
  summary TEXT,
  session_s3_path TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  processed_for_training BOOLEAN DEFAULT FALSE
);
```

## Implementation Details

### L0 Processing (Document Chunking)

- Uses semantic chunking based on topic boundaries
- Handles various document formats (text, PDF, Office)
- Extracts metadata for improved retrieval
- Generates embeddings using OpenAI API
- Stores original document and each chunk as separate files in Wasabi
- Stores chunk embeddings and pointers to chunk files in Weaviate (not the chunk text itself)

### L1 Processing (Hierarchical Memory)

- Creates semantic clusters of related chunks
- Identifies key entities and relationships
- Organizes content into hierarchical structure
- Generates topic labels for clusters
- Stores structure in Wasabi for L2 processing

### L2 Processing (Model Training)

- Generates synthetic training data based on:
  - User preferences extracted from documents
  - Diverse opinions and viewpoints
  - Self-awareness questions and answers
- Fine-tunes Qwen2.5-0.5B-Instruct with LoRA
- Converts to GGUF format for efficient inference
- Handles training failures and retries

### Inference with RAG

- Combines fine-tuned model with retrieved context
- Uses semantic search to find relevant chunks
- Directly retrieves pre-processed chunks from Wasabi using the s3_path pointers
- Formats prompts with retrieved context
- Caches models for improved performance
- Stores new conversations as chat files in Wasabi for future training


## Deployment Considerations

- All Modal functions run in US-West-1 region (same as Wasabi)
- Uses Modal persistent volumes for model caching
- Implements proper error handling and retries
- Logs detailed information for debugging
- Focuses on minimizing costs and latency 

## Chat Storage and Processing

The chat storage approach follows a file-based system that integrates with the training pipeline:

### Chat Object Structure

Conversations are stored as structured objects based on the Chat class from the lpm_kernel:

```python
class Chat:
    def __init__(
        self,
        sessionId: str = "",
        summary: str = "",
        title: str = "",
        createTime: str = "",
    ) -> None:
        self.session_id = sessionId
        self.summary = summary
        self.title = title
        self.create_time = createTime
        self.messages = []  # List of message objects with user/assistant content
```

### Chat Processing Flow

1. **Collection**: When a user interacts with their digital twin, the conversation is captured
2. **Storage**: 
   - The chat session data is stored as JSON files in Wasabi under the chats directory
   - Minimal metadata is tracked in PostgreSQL (session ID, title, path to S3 files)
3. **Processing for Training**:
   - During the L0/L1 processing, chat files are:
     - Chunked and embedded similar to other documents
     - Stored in Weaviate for semantic search
   - During L2 processing, chats are:
     - Loaded as Chat objects
     - Processed into training text files
     - Integrated with other memory types (documents, notes)
     - Used for fine-tuning the model

### Integration with Memory System

Chat histories are treated as a type of "memory" in the system, alongside documents:

```python
class UserInfo:
    def __init__(
        self, cur_time: str, documents: List[Document], chats: List[Chat]
    ):
        # Combine all memory types into a unified memory structure
        self.memories = sorted(
            documents + chats,
            key=lambda x: datetime2timestamp(x.create_time),
            reverse=True,
        )
```

This approach allows the digital twin to learn from conversations as if they were memories, creating a more personalized experience.

### Important Considerations for Training Files

Be careful with mapping training process files to the Wasabi schema, since the training process creates multiple different processed files.

For example, for chat histories, the CURRENT training process in the `lpm_kernel` creates several different types of processed files:

1. **JSON Files**: Initial storage of structured chat data
   - Raw JSON in `L2/data_pipeline/raw_data/`
   - Processed JSON in subjective/objective directories 

2. **Text Files**: Converted from JSON for training
   - Individual text files in `L1/processed_data/subjective/` and `L1/processed_data/objective/`
   - Each chat becomes multiple training examples

3. **GraphRAG Files**: For entity extraction and relationships
   - Entity files in `L1/graphrag_indexing_output/subjective/entities.parquet`
   - Relationship files for semantic context

When adapting to Wasabi storage, we need to maintain this hierarchical structure while ensuring files are properly organized under the tenant ID. Our architecture should:

- Preserve all intermediate processing files
- Maintain clear paths between raw and processed data
- Ensure training scripts can locate all required files
- Track versions of processed data tied to training runs

This may require additional subdirectories in our Wasabi schema beyond what's shown in the S3 layout above.

