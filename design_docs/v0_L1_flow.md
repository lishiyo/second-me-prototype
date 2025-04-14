# L1 Layer Implementation Plan

## Overview

This document outlines the plan for implementing the L1 layer in our digital twin prototype. The L1 layer processes data from the L0 layer to create higher-level knowledge representations, including topics, clusters, shades, and biographies. Our implementation will follow the approach used in `lpm_kernel` but adapt it to our stack of Wasabi, Weaviate, and PostgreSQL.

## Data Flow

```
L0 Documents → Extract Notes & Memory → Generate Topics → Generate Shades → Merge Shades → Generate Global Biography → Store in Wasabi/Weaviate/PostgreSQL
```

## Component Design

### Core Components

1. **L1Manager** (`app/processors/l1/l1_manager.py`)
   - Orchestrates the L1 generation process
   - Coordinates between different generators
   - Handles error recovery and retries
   - Manages data flow between components

2. **L1Generator** (`app/processors/l1/l1_generator.py`)
   - Core logic for generating L1 representations
   - Delegates to specialized generators
   - Provides utility functions for LLM interaction
   - Implements retry logic for LLM calls

3. **TopicsGenerator** (`app/processors/l1/topics_generator.py`)
   - Creates topic clusters from document embeddings
   - Implements clustering algorithms (similar to `lpm_kernel`)
   - Generates topic labels for each cluster
   - Handles outlier documents

4. **ShadeGenerator** (`app/processors/l1/shade_generator.py`)
   - Generates "shades" (knowledge aspects) from document clusters
   - Extracts insights from related documents
   - Creates coherent narratives from document clusters
   - Assigns confidence levels to generated insights

5. **BiographyGenerator** (`app/processors/l1/biography_generator.py`)
   - Creates user biographies from processed data
   - Generates both global and status biographies
   - Implements perspective shifting (first/second/third person)
   - Synthesizes information from all sources

6. **ShadeMerger** (`app/processors/l1/shade_merger.py`)
   - Merges similar shades to create a coherent knowledge representation
   - Identifies overlapping themes
   - Resolves contradictions
   - Generates unified narratives

### Adapter Components

7. **WasabiStorageAdapter** (`app/providers/wasabi_l1_adapter.py`)
   - Handles storage and retrieval of L1 data from Wasabi
   - Implements folder structure for L1 data
   - Manages versioning of L1 data
   - Optimizes for minimal egress costs

8. **WeaviateAdapter** (`app/providers/weaviate_l1_adapter.py`)
   - Handles storage and retrieval of embeddings and relationships
   - Implements specialized queries for L1 operations
   - Manages multi-tenancy for user separation
   - Optimizes for query performance

9. **PostgresAdapter** (`app/providers/postgres_l1_adapter.py`)
   - Handles storage and retrieval of structured L1 data
   - Implements database schema for L1 entities
   - Manages transaction consistency
   - Optimizes for query performance

## Data Models

### Python Classes

1. **Note** (`app/models/note.py`)
   - Represents a processed document with metadata and embeddings
   - Similar to the Note class in `lpm_kernel`
   - Adapted to include references to Wasabi paths

2. **Chunk** (`app/models/chunk.py`)
   - Represents a segment of a document with its own embedding
   - Includes pointers to the original document
   - Stores embeddings for semantic search

3. **Topic** (`app/models/topic.py`)
   - Represents a thematic grouping of related documents
   - Includes a topic name, summary, and related documents
   - Stores an embedding representing the topic center

4. **Cluster** (`app/models/cluster.py`)
   - Represents a group of related documents
   - Includes member documents and their relationships
   - Stores metadata about the cluster (size, coherence, etc.)

5. **Shade** (`app/models/shade.py`)
   - Represents a knowledge aspect extracted from document clusters
   - Includes insights, confidence levels, and source documents
   - Stores relationships to related shades
   - Includes timeline data to track chronological progression of events
   - Supports improvement/updates to existing shades with new information

6. **Bio** (`app/models/bio.py`)
   - Represents a user biography with different perspective views
   - Includes summary and detailed content
   - Stores confidence levels for different aspects
   - Implements methods for perspective shifting

7. **L1GenerationResult** (`app/models/l1_generation_result.py`)
   - Encapsulates the complete result of the L1 generation process
   - Includes bio, clusters, and topics
   - Provides utility methods for result analysis

### Database Schema

#### PostgreSQL Schema

```sql
-- Topics table
CREATE TABLE topics (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  name TEXT NOT NULL,
  summary TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  s3_path TEXT NOT NULL -- Path to detailed data in Wasabi
);

-- Clusters table
CREATE TABLE clusters (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  topic_id UUID REFERENCES topics(id),
  name TEXT,
  summary TEXT,
  document_count INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  s3_path TEXT NOT NULL -- Path to detailed data in Wasabi
);

-- Cluster-Document junction table
CREATE TABLE cluster_documents (
  cluster_id UUID REFERENCES clusters(id),
  document_id UUID REFERENCES documents(id),
  similarity_score FLOAT,
  PRIMARY KEY (cluster_id, document_id)
);

-- Shades table
CREATE TABLE shades (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  name TEXT NOT NULL,
  summary TEXT,
  confidence FLOAT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  s3_path TEXT NOT NULL -- Path to detailed data in Wasabi
);

-- Shade-Cluster junction table
CREATE TABLE shade_clusters (
  shade_id UUID REFERENCES shades(id),
  cluster_id UUID REFERENCES clusters(id),
  PRIMARY KEY (shade_id, cluster_id)
);

-- Global biographies table
CREATE TABLE global_biographies (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  content TEXT NOT NULL,
  content_third_view TEXT NOT NULL,
  summary TEXT NOT NULL,
  summary_third_view TEXT NOT NULL,
  confidence FLOAT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  version INTEGER NOT NULL
);

-- Status biographies table
CREATE TABLE status_biographies (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  content TEXT NOT NULL,
  content_third_view TEXT NOT NULL,
  summary TEXT NOT NULL,
  summary_third_view TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- L1 processing versions
CREATE TABLE l1_versions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  version INTEGER NOT NULL,
  status TEXT NOT NULL, -- 'processing', 'completed', 'failed'
  started_at TIMESTAMP NOT NULL,
  completed_at TIMESTAMP,
  error TEXT,
  UNIQUE (user_id, version)
);
```

#### Weaviate Schema

```python
# TopicClass in Weaviate
{
  "class": "TenantTopic",
  "description": "A topic grouping related documents",
  "properties": [
    {
      "name": "topicId",
      "description": "Unique identifier for the topic",
      "dataType": ["string"]
    },
    {
      "name": "name",
      "description": "Name of the topic",
      "dataType": ["string"]
    },
    {
      "name": "summary",
      "description": "Summary of the topic",
      "dataType": ["text"]
    },
    {
      "name": "documentIds",
      "description": "List of document IDs in this topic",
      "dataType": ["string[]"]
    },
    {
      "name": "created",
      "description": "Creation timestamp",
      "dataType": ["date"]
    }
  ]
}

# ClusterClass in Weaviate
{
  "class": "TenantCluster",
  "description": "A cluster of semantically related documents",
  "properties": [
    {
      "name": "clusterId",
      "description": "Unique identifier for the cluster",
      "dataType": ["string"]
    },
    {
      "name": "topicId",
      "description": "ID of the parent topic",
      "dataType": ["string"]
    },
    {
      "name": "name",
      "description": "Name of the cluster",
      "dataType": ["string"]
    },
    {
      "name": "documentIds",
      "description": "List of document IDs in this cluster",
      "dataType": ["string[]"]
    },
    {
      "name": "created",
      "description": "Creation timestamp",
      "dataType": ["date"]
    }
  ]
}
```

### Wasabi S3 Structure

Building on the structure defined in the architecture document:

```
tenant/1/                                # User ID 1
  ├── ...                                # Existing structure from L0
  ├── training_data/                     # Training data
      ├── L1/                            # L1 processing outputs
          ├── processed_data/            # Processed data files
          │   ├── subjective/            # Subjective training data
          │   │   ├── notes/             # Processed notes
          │   │   │   ├── note_[id].json # Individual note data
          │   │   │   └── ...
          │   │   ├── chats/             # Processed chats
          │   │   │   └── ...
          │   │   └── ...
          │   └── objective/             # Objective training data
          │       └── ...
          ├── topics/                    # Topic data
          │   ├── topic_[id].json        # Individual topic data
          │   └── ...
          ├── clusters/                  # Cluster data
          │   ├── cluster_[id].json      # Individual cluster data
          │   └── ...
          ├── shades/                    # Shade data
          │   ├── shade_[id].json        # Individual shade data
          │   └── ...
          ├── merged_shades/             # Merged shade data
          │   ├── merged_[id].json       # Individual merged shade data
          │   └── ...
          ├── biographies/               # Biography data
          │   ├── global/                # Global biographies
          │   │   ├── bio_[version].json # Version-specific biography
          │   │   └── ...
          │   └── status/                # Status biographies
          │       ├── status_[date].json # Date-specific status
          │       └── ...
          └── graphrag_indexing/         # GraphRAG indexing outputs
              ├── subjective/
              │   ├── entities.parquet
              │   └── relations.parquet
              └── objective/
                  └── ...
```

## Implementation Flow

### 1. Extract Notes from L0 Documents

```python
# In l1_manager.py
async def extract_notes_from_documents(user_id: str) -> List[Note]:
    """Extract Note objects from L0 documents"""
    # Get document IDs from PostgreSQL
    document_ids = await postgres_adapter.get_document_ids_with_l0(user_id)
    
    notes_list = []
    for doc_id in document_ids:
        # Get document metadata from PostgreSQL
        doc_metadata = await postgres_adapter.get_document_metadata(doc_id)
        
        # Get document chunks from Weaviate
        chunk_ids = await weaviate_adapter.get_chunk_ids_by_document(user_id, doc_id)
        chunks = []
        
        for chunk_id in chunk_ids:
            # Get chunk embedding from Weaviate
            chunk_embedding = await weaviate_adapter.get_chunk_embedding(user_id, chunk_id)
            
            # Get chunk content from Wasabi
            chunk_content = await wasabi_adapter.get_chunk_content(user_id, doc_id, chunk_id)
            
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                embedding=chunk_embedding
            ))
        
        # Get document embedding from Weaviate
        doc_embedding = await weaviate_adapter.get_document_embedding(user_id, doc_id)
        
        # Create Note object
        note = Note(
            id=doc_id,
            content=await wasabi_adapter.get_document_content(user_id, doc_id),
            create_time=doc_metadata['uploaded_at'],
            embedding=doc_embedding,
            chunks=chunks,
            title=doc_metadata.get('title', ''),
            summary=doc_metadata.get('summary', {}),
            insight=doc_metadata.get('insight', {})
        )
        
        notes_list.append(note)
    
    return notes_list
```

### 2. Generate Topics and Clusters

```python
# In l1_manager.py
async def generate_topics_and_clusters(user_id: str, notes: List[Note]) -> Dict:
    """Generate topics and clusters from notes"""
    memory_list = [{
        "memoryId": str(note.id),
        "embedding": note.embedding
    } for note in notes]
    
    # Generate topics and clusters
    l1_generator = L1Generator()
    topics_generator = TopicsGenerator()
    
    # Get existing clusters from previous runs (if any)
    old_clusters = await postgres_adapter.get_latest_clusters(user_id)
    old_outliers = await postgres_adapter.get_latest_outliers(user_id)
    
    # Generate new clusters
    clusters = topics_generator.generate_topics_for_shades(
        old_cluster_list=old_clusters,
        old_outlier_memory_list=old_outliers,
        new_memory_list=memory_list
    )
    
    # Generate chunk topics
    chunk_topics = l1_generator.generate_topics(notes)
    
    # Store clusters in Wasabi
    for cluster in clusters.get("clusterList", []):
        cluster_id = str(uuid.uuid4())
        await wasabi_adapter.store_cluster(
            user_id=user_id,
            cluster_id=cluster_id,
            cluster_data=cluster
        )
        
        # Store in PostgreSQL
        await postgres_adapter.create_cluster(
            user_id=user_id,
            cluster_id=cluster_id,
            topic_id=cluster.get("topicId"),
            name=cluster.get("name", ""),
            summary=cluster.get("summary", ""),
            document_count=len(cluster.get("memoryList", [])),
            s3_path=f"tenant/{user_id}/training_data/L1/clusters/cluster_{cluster_id}.json"
        )
        
        # Store in Weaviate
        await weaviate_adapter.create_cluster(
            user_id=user_id,
            cluster_id=cluster_id,
            topic_id=cluster.get("topicId"),
            name=cluster.get("name", ""),
            document_ids=[m.get("memoryId") for m in cluster.get("memoryList", [])],
            embedding=cluster.get("centerEmbedding")
        )
    
    return {
        "clusters": clusters,
        "chunk_topics": chunk_topics
    }
```

### 3. Generate Shades

```python
# In l1_manager.py
async def generate_shades(user_id: str, notes: List[Note], clusters: Dict) -> List[Dict]:
    """Generate shades from clusters"""
    l1_generator = L1Generator()
    shades = []
    
    for cluster in clusters.get("clusterList", []):
        cluster_memory_ids = [str(m.get("memoryId")) for m in cluster.get("memoryList", [])]
        cluster_notes = [note for note in notes if str(note.id) in cluster_memory_ids]
        
        if cluster_notes:
            # Get existing shades (if any)
            old_shades = await postgres_adapter.get_shade_info_by_cluster(user_id, cluster.get("id"))
            
            # Generate shade for this cluster
            shade = l1_generator.gen_shade_for_cluster(
                old_memory_list=[],
                new_memory_list=cluster_notes,
                shade_info_list=old_shades
            )
            
            if shade:
                shade_id = str(uuid.uuid4())
                shade["id"] = shade_id
                
                # Store timeline data in shade metadata
                if "timeline" in shade:
                    # Ensure timeline data is properly stored
                    pass
                
                # Store shade in Wasabi
                await wasabi_adapter.store_shade(
                    user_id=user_id,
                    shade_id=shade_id,
                    shade_data=shade
                )
                
                # Store in PostgreSQL
                await postgres_adapter.create_shade(
                    user_id=user_id,
                    shade_id=shade_id,
                    name=shade.get("name", ""),
                    summary=shade.get("summary", ""),
                    confidence=shade.get("confidence", 0.0),
                    s3_path=f"tenant/{user_id}/training_data/L1/shades/shade_{shade_id}.json"
                )
                
                # Link shade to cluster
                await postgres_adapter.link_shade_to_cluster(shade_id, cluster.get("id"))
                
                shades.append(shade)
    
    return shades
```

### 4. Merge Shades

```python
# In l1_manager.py
async def merge_shades(user_id: str, shades: List[Dict]) -> Dict:
    """Merge similar shades"""
    l1_generator = L1Generator()
    
    # Merge shades
    merged_result = l1_generator.merge_shades(shades)
    
    if merged_result.success:
        # Store merged shades in Wasabi
        for merged_shade in merged_result.merge_shade_list:
            merged_id = str(uuid.uuid4())
            
            # Ensure timeline data is preserved in merged shades
            if "timeline" in merged_shade:
                # Process and store timeline data
                pass
            
            await wasabi_adapter.store_merged_shade(
                user_id=user_id,
                merged_id=merged_id,
                merged_data=merged_shade
            )
    
    return merged_result
```

### 5. Generate Global Biography

```python
# In l1_manager.py
async def generate_global_biography(user_id: str, merged_shades: Dict, clusters: Dict) -> Dict:
    """Generate global biography"""
    l1_generator = L1Generator()
    
    # Get existing profile (if any)
    old_profile = await postgres_adapter.get_latest_global_bio(user_id)
    if not old_profile:
        old_profile = Bio(shadesList=merged_shades.merge_shade_list if merged_shades.success else [])
    
    # Generate global biography
    bio = l1_generator.gen_global_biography(
        old_profile=old_profile,
        cluster_list=clusters.get("clusterList", [])
    )
    
    # Get next version number
    version = await postgres_adapter.get_next_bio_version(user_id)
    
    # Store in Wasabi
    await wasabi_adapter.store_global_bio(
        user_id=user_id,
        version=version,
        bio_data=bio.to_dict()
    )
    
    # Store in PostgreSQL
    await postgres_adapter.create_global_bio(
        user_id=user_id,
        bio_id=str(uuid.uuid4()),
        content=bio.content_second_view,
        content_third_view=bio.content_third_view,
        summary=bio.summary_second_view,
        summary_third_view=bio.summary_third_view,
        confidence=bio.confidence,
        version=version
    )
    
    return bio
```

### 6. Generate Status Biography

```python
# In l1_manager.py
async def generate_status_biography(user_id: str, notes: List[Note]) -> Dict:
    """Generate status biography"""
    l1_generator = L1Generator()
    
    # Generate status biography
    current_time = datetime.now().strftime("%Y-%m-%d")
    status_bio = l1_generator.gen_status_biography(
        cur_time=current_time,
        notes=notes,
        todos=[],  # Empty for now
        chats=[]   # Empty for now
    )
    
    # Store in Wasabi
    await wasabi_adapter.store_status_bio(
        user_id=user_id,
        date=current_time,
        bio_data=status_bio.to_dict()
    )
    
    # Store in PostgreSQL
    await postgres_adapter.create_status_bio(
        user_id=user_id,
        bio_id=str(uuid.uuid4()),
        content=status_bio.content_second_view,
        content_third_view=status_bio.content_third_view,
        summary=status_bio.summary_second_view,
        summary_third_view=status_bio.summary_third_view
    )
    
    return status_bio
```

### 7. Complete L1 Generation Process

```python
# In l1_manager.py
async def generate_l1_from_l0(user_id: str) -> L1GenerationResult:
    """Generate L1 level knowledge representation from L0 data"""
    try:
        # Create a new L1 version
        version = await postgres_adapter.create_l1_version(user_id)
        
        # 1. Extract notes from L0 documents
        notes = await extract_notes_from_documents(user_id)
        
        # 2. Generate topics and clusters
        topic_result = await generate_topics_and_clusters(user_id, notes)
        
        # 3. Generate shades for each cluster
        shades = await generate_shades(user_id, notes, topic_result["clusters"])
        
        # 4. Merge shades
        merged_shades = await merge_shades(user_id, shades)
        
        # 5. Generate global biography
        bio = await generate_global_biography(user_id, merged_shades, topic_result["clusters"])
        
        # 6. Generate status biography
        status_bio = await generate_status_biography(user_id, notes)
        
        # 7. Build result object
        result = L1GenerationResult(
            bio=bio,
            clusters=topic_result["clusters"],
            chunk_topics=topic_result["chunk_topics"]
        )
        
        # 8. Update L1 version status
        await postgres_adapter.complete_l1_version(user_id, version)
        
        return result
    except Exception as e:
        logger.error(f"Error in L1 generation: {str(e)}", exc_info=True)
        await postgres_adapter.fail_l1_version(user_id, version, str(e))
        raise
```

## Performance Considerations

1. **Asynchronous Processing**
   - Use async/await patterns for all I/O operations
   - Implement concurrent processing where possible
   - Batch operations to reduce API calls

2. **Caching Strategies**
   - Cache frequently accessed data in memory
   - Implement TTL-based caching for embeddings
   - Cache document content during processing

3. **Efficient Embedding Operations**
   - Batch embedding operations
   - Use approximate nearest neighbors for clustering
   - Implement incremental updates for topics

4. **Optimized Storage**
   - Store full data in Wasabi, metadata in PostgreSQL
   - Use compression for stored JSON files
   - Implement efficient retrieval patterns

5. **LLM Optimization**
   - Implement retry logic with backoff
   - Implement parameter adjustment strategy (especially top_p) for failed LLM calls
   - Optimize prompt templates for token efficiency
   - Cache LLM results where appropriate
   - Support for updating and improving existing shades with new data

## Potential Challenges and Risks

1. **Data Consistency**
   - Challenge: Ensuring consistency across three storage systems (Wasabi, Weaviate, PostgreSQL)
   - Mitigation: Implement transactional semantics and rollback mechanisms

2. **Performance at Scale**
   - Challenge: Processing large numbers of documents efficiently
   - Mitigation: Implement incremental processing and parallelization

3. **Cost Management**
   - Challenge: Managing costs of storage and API calls
   - Mitigation: Optimize data storage, implement caching, batch operations

4. **Error Handling**
   - Challenge: Gracefully handling failures at different stages
   - Mitigation: Implement comprehensive error handling and retry mechanisms

5. **LLM Reliability**
   - Challenge: Handling LLM API failures and rate limits
   - Mitigation: Implement backoff strategies and fallback mechanisms
   - Dynamic parameter adjustment for failed calls (adjusting top_p, temperature, etc.)
   - Provide detailed error messages for debugging API failures

6. **Versioning Conflicts**
   - Challenge: Managing updates to previously processed data
   - Mitigation: Implement proper versioning and change tracking

7. **Multi-tenancy**
   - Challenge: Properly isolating data between users
   - Mitigation: Enforce strict user_id scoping in all operations

## Testing Strategy

1. **Unit Tests**
   - Test individual components with mocked dependencies
   - Verify correct handling of edge cases
   - Ensure proper error handling

2. **Integration Tests**
   - Test interaction between components
   - Verify data flow through the system
   - Test with realistic data volumes

3. **End-to-End Tests**
   - Test complete L1 generation process
   - Verify results against expected outputs
   - Test recovery from failures

## Implementation Plan

1. **Phase 1: Core Infrastructure**
   - Implement data models
   - Set up storage adapters
   - Create database schema

2. **Phase 2: Basic L1 Generation**
   - Implement topic clustering
   - Implement shade generation
   - Implement basic biography generation

3. **Phase 3: Enhanced Features**
   - Implement shade merging
   - Implement perspective shifting
   - Implement confidence assignment

4. **Phase 4: Optimization**
   - Implement caching strategies
   - Optimize performance
   - Enhance error handling

5. **Phase 5: Testing and Refinement**
   - Comprehensive testing
   - Performance tuning
   - Documentation

## Conclusion

This implementation plan adapts the L1 layer from `lpm_kernel` to our stack of Wasabi, Weaviate, and PostgreSQL. By following this plan, we can create a robust L1 layer that processes L0 data into higher-level knowledge representations, building on our existing L0 implementation. The plan addresses key performance considerations and potential challenges, providing a clear path forward for implementation. 