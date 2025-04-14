# L1 Data Models

This directory contains the data models for the L1 layer of our digital twin prototype. These models represent higher-level knowledge representations derived from L0 document data.

## Core Models

### Note

`Note` represents a processed document with its metadata and embeddings. It serves as the input to the L1 generation process, extracted from L0 documents.

Key attributes:
- `id`: Unique identifier
- `content`: Document content
- `create_time`: Creation timestamp
- `embedding`: Document embedding
- `chunks`: List of document chunks
- `title`: Document title
- `summary`: Document summary
- `insight`: Document insight
- `tags`: List of document tags

### Chunk

`Chunk` represents a segment of a document with its own embedding. It's used for fine-grained content analysis.

Key attributes:
- `id`: Unique identifier
- `content`: Chunk content
- `embedding`: Chunk embedding
- `document_id`: Parent document ID
- `metadata`: Additional metadata

## Organizational Models

### Topic

`Topic` represents a thematic grouping of related documents. It provides a high-level categorization of document content.

Key attributes:
- `id`: Unique identifier
- `name`: Topic name
- `summary`: Topic summary
- `document_ids`: List of document IDs in this topic
- `embedding`: Topic embedding centroid

### Memory

`Memory` represents a reference to a document in the context of clustering. It's used for document similarity analysis.

Key attributes:
- `memory_id`: Document ID
- `embedding`: Document embedding
- `metadata`: Additional metadata

### Cluster

`Cluster` represents a group of related documents clustered by similarity. It's a more specific grouping than a Topic.

Key attributes:
- `id`: Unique identifier
- `topic_id`: Parent topic ID
- `name`: Cluster name
- `summary`: Cluster summary
- `memory_list`: List of memories in this cluster
- `center_embedding`: Cluster centroid embedding

## Knowledge Models

### Shade

`Shade` represents a knowledge aspect extracted from document clusters. It contains insights derived from related documents.

Key attributes:
- `id`: Unique identifier
- `name`: Shade name
- `summary`: Shade summary
- `content`: Detailed content
- `confidence`: Confidence score
- `source_clusters`: List of source cluster IDs

### ShadeInfo

`ShadeInfo` contains information about existing shades for a cluster. It's used when generating or updating shades.

Key attributes:
- `shade_id`: Shade ID
- `name`: Shade name
- `content`: Shade content
- `confidence`: Confidence score

### ShadeMergeInfo

`ShadeMergeInfo` contains information about shades to be merged. It's used in the shade merging process.

Key attributes:
- `shade_id`: Shade ID
- `name`: Shade name
- `summary`: Shade summary
- `content`: Shade content
- `confidence`: Confidence score
- `source_clusters`: List of source cluster IDs

### MergedShadeResult

`MergedShadeResult` represents the result of a shade merging operation. It contains the merged shades or error information.

Key attributes:
- `success`: Whether the operation was successful
- `merge_shade_list`: List of merged shades
- `error`: Error information, if any

## Biographical Models

### Bio

`Bio` represents a user biography with different perspective views. It synthesizes information from shades into a coherent narrative.

Key attributes:
- `content_first_view`: First-person biography content
- `summary_first_view`: First-person biography summary
- `content_second_view`: Second-person biography content
- `summary_second_view`: Second-person biography summary
- `content_third_view`: Third-person biography content
- `summary_third_view`: Third-person biography summary
- `confidence`: Overall confidence score
- `shades_list`: List of shades used to generate the biography

## Result Models

### L1GenerationResult

`L1GenerationResult` encapsulates the complete result of the L1 generation process. It contains the generated bio, clusters, and topics.

Key attributes:
- `bio`: Generated biography
- `clusters`: Generated clusters
- `chunk_topics`: Generated chunk topics
- `status`: Operation status
- `error`: Error information, if any 