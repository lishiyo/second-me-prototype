# MVP overview

We are building a "digital twin" prototype that takes in a bunch of any user's documents (and in the future other kinds of data like tweets, chat logs, voting behavior) and creates a chatbot that can both interact with them and represent them as an agent to other chatbots. This is done through using a vector DB with GraphRAG as well as finetuning a base model to represent their identity.

The main flow entails:
- For now, we are only using one user (`user_id: 1`). We are not implementing authentication or authorization yet, just assume all the data here is for `user_id: 1`.
- Document Storage: the full files in `data` directory (the docs for the user `user_id: 1`) should be uploaded to Wasabi
    - future: we will create a web interface to let users sign in and upload documents themselves
    - the documents will be limited to max 100 files (max 10MB each) for now
- Vector Storage: chunk and embed the files to Weaviate Cloud, which should have chunk_index, byte_start, and byte_end pointers to fetch the chunk text from Wasabi. This enables semantic search during conversations.
- Training process to train the digital twin to reflect your identity and context (see `lpm_kernel` directory, which contains the example logic we want):
    - Download the base model (`Qwen2.5-0.5B-Instruct`)
    - Organize the data into a hierarchical structure via GraphRAG, combining the embeddings with metadata to prioritize certain memories
    - Synthesize training data based on the user's identity and documents
    - Fine-tune the model using LoRA (Low-Rank Adaptation)
     - Convert to efficient GGUF format for inference
- Create the chatbot (the digital twin) to interact with the user:
    - the digital twin searches the vector db for relevant chunks, uses the retrieved chunks + fine-tuned behavior to generate a response
    - the interaction is stored as a new memory (chunked + embedded) for future use
    - in the long-term, on a periodical basis (once a week), we will batch the new data and retrain the model - do NOT worry about this for MVP

Notes for MVP:
- do not worry about authentication or authorization, assume `user_id: 1` for everything
- do not worry about retraining, we will add that later
- do not worry about sensitive information in the training data for now
- for failed training jobs, log and attempt to retry up to 3 times
- for excessive usage, do not worry about that yet - we will limit to 100 documents for now


## Tech Stack

Wasabi - S3 file storage, store full documents in `raw_content` column and chunks in another table, each must be scoped under the user_id
PostgreSQL - store info about the user, other structured data
    - this should have row-level security
Weaviate 4 Cloud with multi-tenancy - store embeddings and pointers to the chunks, use for semantic searches
Modal serverless job (`modal.Function`) - use for training and inference
FastAPI - backend api, use Redis for queue

We will use OpenAI's `text-embedding-3-small` for the embeddings model, and `gpt-4o-mini` for chat completions.

The base model we are using to finetune is `Qwen2.5-0.5B-Instruct`. IMPORTANT: Refer to the code in `lpm_kernel` directory for example code of this training flow, we want to re-use as much of this as we can:
- the `models` directory contains the implementation of the Hierarchical Memory Modeling
- the `L0`, `L1`, `L2` directory represent hierachical layers of the memory or training pipeline:
    - `L0` does the initial data preprocessing
    - `L1` transforms chunked data into hierarchical memory structure
        - `lpm_kernel/L1/topics_generator.py` creates "topics" (thematic groupings) from memories and organizes them into semantic clusters
    - `L2` prepares processed data for model training, downloads the model, and controls the fine-tuning process
        - `lpm_kernel/L2/mlx_training/` provides optimized training for Apple Silicon M-series chips.
    - IMPORTANT: we need to modify this code for our purposes since it assumes local disk storage for everything. Make sure L0 and L1 write to Wasabi (raw) + Weaviate (vector) instead of `resources/...`. 

Model Update Process
- Data Ingestion: New data (memories, chats, etc.) is first chunked by L0.
- Hierarchical Processing: L1 organizes chunks into semantic structures.
- Training Data Preparation: L2 prepares data for model updates.
- Fine-tuning: L2's training pipeline updates the model with new user data.

## Sample directory structure

```
second-me-prototype/
├── app/                       # FastAPI service
│   ├── main.py
│   ├── api/                   # routers
│   │   ├── ingest.py
│   │   └── chat.py
│   ├── core/                  # auth, settings
│   │   ├── config.py
│   │   └── dependencies.py
│   ├── db/                    # Postgres SQLModel models
│   │   └── models.py
│   └── providers/             # pluggable back‑ends
│       ├── blob_store.py
│       ├── vector_db.py
│       └── rel_db.py
├── worker/                    # Redis-backed background workers
│   ├── ingest_worker.py
│   └── retrain_worker.py
├── training/
│   ├── train_modal.py         
│   ├── image/                 # Optional custom image (Dockerfile)
├── inference/
│   └── serve_modal.py             
├── lpm_kernel/                # example training code
│   └── ...
└── pyproject.toml             # Poetry 
```

## Training & inference flow

We want to use Modal for the training and inference. Here is a SUGGESTED implementation strategy, you can refine it:

Containerise once:

Create one Dockerfile that:
- Downloads the base model (`qwen2.5‑0.5b-instruct`).
- Mounts /data (Modal persistent volume).
- Runs your existing training script.

We pass a `--resume-from-step` flag so retraining only re‑runs the new data chunk.

Artifact flow
1. Job writes LoRA weights to /tmp/out/adapter.safetensors.
2. Script uploads to `blob_store.put(key=f"tenant/{id}/lora/{hash}.safetensors")`.
3. API stores the blob key in Postgres (`lora_path`).
4. Inference container periodically checks Postgres for “model_ready = true” and hot‑reloads.

Inference service
For 0.5 B GGUF we can serve all tenants in one llama.cpp process on a single‑GPU instance (A10g has 24 GB). The process keeps:
- Base model in VRAM.
- On request, merges tenant LoRA with apply_lora() into RAM; cache the last N tenants.

Training and Inference flow:
- Training job streams raw docs from `wasabi://tenant/<id>/raw/....`
- Outputs adapter.safetensors → `wasabi://tenant/<id>/lora/<hash>.safetensors`
- Inference service downloads from Wasabi on cold start and caches in `/tmp`
- The retrain endpoint stores the artifact key in Postgres (`lora_path`)


Calling Modal from our FastAPI backend should be simple:

```python
from modal import App
app = App.from_name("second-me-serve")      # loads deployed inference app

async def chat_with_twin(tenant_id: str, prompt: str):
    # non‑blocking HTTP call under the hood
    return await app.functions.generate.call_async(tenant_id, prompt)
```

## Implementation steps

1. Create a detailed architectural diagram showing all components and data flows
2. Implement the core adapters for Wasabi, Weaviate, and PostgreSQL
3. Build the L0 processing pipeline first, with proper testing
4. Incrementally add L1 and L2 processing
5. Develop the inference service with RAG
6. Build the FastAPI backend and integrate with Modal
7. Implement the Redis-backed worker system


## Operational tips

Keep Wasabi egress‑free by running both Modal in us‑west‑1 (same as your bucket).

Cache models in Modal Volume to avoid paying for repeated downloads.

Use Weaviate’s multi‑tenant shards to fetch context inside your inference function (same code on both platforms).

Wire training completion → Postgres update → Webhook that triggers a “hot reload” in the inference service.

Set a max concurrency per tenant (e.g., 2) so one user can’t starve GPUs.