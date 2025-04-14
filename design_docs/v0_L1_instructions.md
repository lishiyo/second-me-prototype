# L1 Layer Implementation Instructions

## Introduction

This document provides detailed implementation instructions for the L1 layer of our digital twin prototype. It breaks down the implementation plan from `v0_L1_flow.md` into specific subtasks with step-by-step instructions and clear dependencies.

The L1 layer processes data from the L0 layer to create higher-level knowledge representations, including topics, clusters, shades, and biographies. Our implementation follows the approach used in `lpm_kernel` but adapts it to our stack of Wasabi, Weaviate, and PostgreSQL.

### High-Level Dependency Graph

```
                 ┌─────────────┐
                 │  Data Models│
                 └──────┬──────┘
                        │
                        ▼
            ┌───────────────────────┐
            │    Storage Adapters   │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │     L1Generator      │
            └───────────┬───────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│TopicsGenerator  │───────▶ │ ShadeGenerator  │
└─────────────────┘         └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  ShadeMerger    │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │BiographyGenerator│
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   L1Manager     │
                            └─────────────────┘
```

### Implementation Phases

1. **Foundation Phase**: Set up data models, database schemas, and storage adapters
2. **Core Processing Phase**: Implement basic generators and extraction from L0
3. **Advanced Processing Phase**: Implement shade merging and biography generation
4. **Integration Phase**: Connect all components and implement end-to-end flow
5. **Testing and Optimization Phase**: Implement tests and optimize performance

## Phase 1: Foundation Phase

### Subtask 1.1: Implement Data Models

**Description**: Create Python classes for all data models needed in the L1 layer.

**Dependencies**: None

**Steps**:

1. Create a new directory `app/models/l1` for L1-specific models
2. Implement the `Note` class:
   - Include fields for id, content, create_time, embedding, chunks, title, summary, insight
   - Add methods for serialization/deserialization
3. Implement the `Chunk` class:
   - Include fields for id, content, embedding, document_id
   - Add methods for serialization/deserialization
4. Implement the `Topic` class:
   - Include fields for id, name, summary, document_ids, embedding
   - Add methods for serialization/deserialization
5. Implement the `Cluster` class:
   - Include fields for id, topic_id, name, summary, document_ids, center_embedding
   - Add methods for serialization/deserialization
6. Implement the `Shade` class:
   - Include fields for id, name, summary, confidence, source_clusters, content
   - Add methods for serialization/deserialization
7. Implement the `Bio` class:
   - Include fields for content_first_view, content_second_view, content_third_view
   - Include fields for summary_first_view, summary_second_view, summary_third_view
   - Add methods for perspective shifting
   - Add methods for serialization/deserialization
8. Implement the `L1GenerationResult` class:
   - Include fields for bio, clusters, chunk_topics
   - Add utility methods for result analysis

**Definition of Done**:
- All model classes implemented with proper type hints
- Unit tests for each model class
- Documentation for each model class

### Subtask 1.2: Set Up Database Schema

**Description**: Create and apply database schema for PostgreSQL and Weaviate.

**Dependencies**: Subtask 1.1

**Steps**:

1. Create SQL migration scripts for PostgreSQL:
   - Create `topics` table
   - Create `clusters` table
   - Create `cluster_documents` junction table
   - Create `shades` table
   - Create `shade_clusters` junction table
   - Create `global_biographies` table
   - Create `status_biographies` table
   - Create `l1_versions` table
2. Set up PostgreSQL schema using SQLModel:
   - Create model classes that correspond to the tables
   - Set up relationships between models
3. Create Weaviate schema definition:
   - Define `TenantTopic` class
   - Define `TenantCluster` class
   - Ensure multi-tenancy is properly configured
4. Implement schema application scripts:
   - Create script to apply PostgreSQL schema
   - Create script to apply Weaviate schema

**Definition of Done**:
- SQL migration scripts created and tested
- SQLModel classes implemented
- Weaviate schema defined
- Schema application scripts tested

### Subtask 1.3: Implement Storage Adapters

**Description**: Create adapter classes for interacting with Wasabi, Weaviate, and PostgreSQL.

**Dependencies**: Subtask 1.2

**Steps**:

1. Implement `WasabiStorageAdapter` class:
   - Implement methods for storing/retrieving Note objects
   - Implement methods for storing/retrieving Cluster objects
   - Implement methods for storing/retrieving Shade objects
   - Implement methods for storing/retrieving Bio objects
   - Ensure proper multi-tenancy by using user_id in paths
2. Implement `WeaviateAdapter` class:
   - Implement methods for storing/retrieving Topic embeddings
   - Implement methods for storing/retrieving Cluster embeddings
   - Implement methods for semantic search across embeddings
   - Ensure proper multi-tenancy using tenant prefix
3. Implement `PostgresAdapter` class:
   - Implement methods for storing/retrieving Topic metadata
   - Implement methods for storing/retrieving Cluster metadata
   - Implement methods for storing/retrieving Shade metadata
   - Implement methods for storing/retrieving Bio metadata
   - Implement methods for versioning and status tracking
   - Ensure proper multi-tenancy using user_id filter

**Definition of Done**:
- All adapter classes implemented with proper error handling
- Asynchronous I/O used for all operations
- Multi-tenancy properly enforced
- Unit tests for each adapter
- Integration tests for adapter interactions

## Phase 2: Core Processing Phase

### Subtask 2.1: Implement L1Generator Base Class

**Description**: Create the core L1Generator class that will be the foundation for specialized generators.

**Dependencies**: Subtask 1.3

**Steps**:

1. Implement the `L1Generator` class:
   - Add constructor with configurable parameters
   - Implement LLM interaction methods
   - Implement retry logic for API calls
   - Add utility methods for common operations
2. Implement caching mechanisms:
   - Add in-memory cache for frequent operations
   - Implement TTL-based cache expiration
3. Set up logging and monitoring:
   - Implement detailed logging for generator operations
   - Add performance metrics collection

**Definition of Done**:
- L1Generator class implemented with proper error handling
- LLM interaction methods tested with mock responses
- Caching mechanisms verified
- Logging and monitoring in place

### Subtask 2.2: Implement Extract Notes from L0

**Description**: Implement functionality to extract Note objects from L0 documents.

**Dependencies**: Subtask 2.1

**Steps**:

1. Implement the `extract_notes_from_documents` function:
   - Get document IDs from PostgreSQL
   - Get document metadata from PostgreSQL
   - Get document chunks from Weaviate
   - Get document content from Wasabi
   - Get document embeddings from Weaviate
   - Create Note objects with all collected data
2. Optimize for large document sets:
   - Implement batch processing
   - Add pagination for large result sets
   - Implement parallel processing where possible

**Definition of Done**:
- Extract function implemented and tested with real L0 data
- Performance benchmarks for different document set sizes
- Error handling for missing or corrupt documents

### Subtask 2.3: Implement Topics Generator

**Description**: Implement the TopicsGenerator class for clustering documents.

**Dependencies**: Subtask 2.1

**Steps**:

1. Implement the `TopicsGenerator` class:
   - Add constructor with configurable parameters
   - Implement clustering algorithm
   - Implement methods for handling outliers
   - Implement methods for merging similar clusters
2. Implement the `generate_topics_for_shades` method:
   - Process old and new clusters
   - Create hierarchical topic structure
   - Assign meaningful names to topics
3. Implement the `generate_topics` method for chunks:
   - Process document chunks
   - Create topic clusters from chunks
   - Generate topic summaries

**Definition of Done**:
- TopicsGenerator class implemented with configurable parameters
- Clustering algorithm tested with sample data
- Topic generation verified for quality and coherence

### Subtask 2.4: Implement Shade Generator

**Description**: Implement the ShadeGenerator class for extracting knowledge aspects.

**Dependencies**: Subtask 2.3

**Steps**:

1. Implement the `ShadeGenerator` class:
   - Add constructor with configurable parameters
   - Implement LLM prompts for shade generation
   - Add methods for confidence scoring
2. Implement the `generate_shade` method:
   - Process cluster notes
   - Generate coherent narratives from related documents
   - Assign confidence levels to generated insights
3. Integrate with Topics Generator:
   - Use cluster information to inform shade generation
   - Ensure consistent naming and organization

**Definition of Done**:
- ShadeGenerator class implemented with configurable parameters
- Shade generation tested with sample clusters
- Quality of generated shades evaluated

## Phase 3: Advanced Processing Phase

### Subtask 3.1: Implement Shade Merger

**Description**: Implement the ShadeMerger class for combining similar shades.

**Dependencies**: Subtask 2.4

**Steps**:

1. Implement the `ShadeMerger` class:
   - Add constructor with configurable parameters
   - Implement similarity detection between shades
   - Add methods for resolving contradictions
2. Implement the `merge_shades` method:
   - Process multiple shades
   - Identify overlapping themes
   - Create unified narratives
   - Assign confidence levels to merged insights

**Definition of Done**:
- ShadeMerger class implemented with configurable parameters
- Shade merging tested with sample shades
- Quality of merged shades evaluated

### Subtask 3.2: Implement Biography Generator

**Description**: Implement the BiographyGenerator class for creating user biographies.

**Dependencies**: Subtask 3.1

**Steps**:

1. Implement the `BiographyGenerator` class:
   - Add constructor with configurable parameters
   - Implement LLM prompts for biography generation
   - Add methods for perspective shifting
2. Implement the `gen_global_biography` method:
   - Process merged shades
   - Generate third-person perspective
   - Convert to second-person perspective
   - Assign confidence levels
3. Implement the `gen_status_biography` method:
   - Process recent notes and activities
   - Generate current status summary
   - Create multiple perspective views

**Definition of Done**:
- BiographyGenerator class implemented with configurable parameters
- Biography generation tested with sample data
- Quality of generated biographies evaluated
- Perspective shifting verified

## Phase 4: Integration Phase

### Subtask 4.1: Implement L1Manager

**Description**: Implement the L1Manager class to orchestrate the entire L1 generation process.

**Dependencies**: Subtasks 3.1, 3.2

**Steps**:

1. Implement the `L1Manager` class:
   - Add constructor with references to all generators and adapters
   - Implement version management
   - Add methods for error recovery
2. Implement the `generate_l1_from_l0` method:
   - Coordinate the complete L1 generation process
   - Manage versioning and status tracking
   - Handle errors and implement retries
3. Implement additional utility methods:
   - Add methods for result retrieval
   - Add methods for status checking
   - Add methods for cleanup and maintenance

**Definition of Done**:
- L1Manager class implemented with proper error handling
- End-to-end process tested with sample data
- Error recovery mechanisms verified

### Subtask 4.2: Implement End-to-End Flow

**Description**: Connect all components to create a complete L1 generation flow.

**Dependencies**: Subtask 4.1

**Steps**:

1. Implement the main L1 generation script:
   - Set up all required components
   - Configure parameters
   - Execute the generation process
   - Report results
2. Implement background worker integration:
   - Add Redis task queue integration
   - Implement worker process for L1 generation
   - Add monitoring and reporting
3. Implement API endpoints:
   - Add endpoint for triggering L1 generation
   - Add endpoints for retrieving results
   - Add endpoints for checking status

**Definition of Done**:
- End-to-end flow implemented and tested
- Background worker integration verified
- API endpoints tested and documented

In particular, we need an api endpoint that makes the same request as lpm_kernel's `POST /api/kernel/l1/global`. This should execute the `generate_l1_from_l0` functionality and return the generated L1 data (bio, clusters, chunk\topics) in a serialized format. We will examine this JSON to verify the structure and content.


## Phase 5: Testing and Optimization Phase

### Subtask 5.1: Implement Unit Tests

**Description**: Create comprehensive unit tests for all components.

**Dependencies**: All Phase 4 subtasks

**Steps**:

1. Implement unit tests for data models:
   - Test serialization/deserialization
   - Test validation logic
   - Test utility methods
2. Implement unit tests for generators:
   - Test with mock dependencies
   - Test core logic in isolation
   - Test error handling
3. Implement unit tests for adapters:
   - Test with mock storage backends
   - Test error handling
   - Test multi-tenancy

**Definition of Done**:
- Unit tests implemented for all components
- Test coverage above 80%
- All tests passing

### Subtask 5.2: Implement Integration Tests

**Description**: Create integration tests for component interactions.

**Dependencies**: Subtask 5.1

**Steps**:

1. Implement integration tests for storage interactions:
   - Test data flow between PostgreSQL, Weaviate, and Wasabi
   - Test consistency across storage systems
   - Test rollback mechanisms
2. Implement integration tests for generator interactions:
   - Test data flow between generators
   - Test end-to-end generation process
3. Implement integration tests for API interactions:
   - Test API endpoints
   - Test background worker integration

**Definition of Done**:
- Integration tests implemented for all component interactions
- Tests using real (test) storage backends
- All tests passing

### Subtask 5.3: Performance Optimization

**Description**: Optimize performance of the L1 generation process.

**Dependencies**: Subtask 5.2

**Steps**:

1. Implement performance benchmarks:
   - Define key performance metrics
   - Create benchmark scripts
   - Establish baseline performance
2. Optimize storage interactions:
   - Implement batch operations
   - Optimize query patterns
   - Add caching where appropriate
3. Optimize generator operations:
   - Implement parallel processing
   - Optimize LLM prompt efficiency
   - Add result caching
4. Implement incremental processing:
   - Add support for processing only new documents
   - Implement differential updates to existing structures
   - Add smart versioning to minimize redundant work

**Definition of Done**:
- Performance benchmarks showing improvement over baseline
- Optimizations implemented and tested
- Documentation updated with performance considerations

## Summary

This implementation plan breaks down the L1 layer development into specific subtasks with clear dependencies and step-by-step instructions. Follow the phases in order, completing all subtasks within each phase before moving to the next.

Key considerations throughout implementation:

1. **Multi-tenancy**: Ensure all components properly isolate data by user_id
2. **Distributed storage**: Manage data consistency across Wasabi, Weaviate, and PostgreSQL
3. **Asynchronous processing**: Use async/await patterns for all I/O operations
4. **Error handling**: Implement robust recovery mechanisms
5. **Testing**: Maintain comprehensive test coverage
6. **Performance**: Optimize for efficient processing of large document sets

By following this plan, we will create a robust L1 layer that processes L0 data into higher-level knowledge representations, building on our existing L0 implementation. 