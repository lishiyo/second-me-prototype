"""
Weaviate schema definitions for L1 data.
This module contains the schema configurations for Weaviate collections used in L1.
Important: We are using Weaviate v4 client, with multi-tenancy enabled.
"""

from weaviate.classes.config import Configure, Property, DataType

# Topic class schema
TENANT_TOPIC_SCHEMA = {
    "name": "TenantTopic",
    "description": "A topic grouping related documents",
    "vectorizer": Configure.Vectorizer.none(),  # We'll provide our own vectors
    "multi_tenancy": Configure.multi_tenancy(
        enabled=True,
        auto_tenant_creation=True,
        auto_tenant_activation=True
    ),
    "properties": [
        Property(
            name="topic_id",
            description="Unique identifier for the topic",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="name",
            description="Name of the topic",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="summary",
            description="Summary of the topic",
            data_type=DataType.TEXT,
            indexing={"filterable": False, "searchable": True}
        ),
        Property(
            name="document_ids",
            description="List of document IDs in this topic",
            data_type=DataType.TEXT_ARRAY,
            indexing={"filterable": True, "searchable": False}
        ),
        Property(
            name="created",
            description="Creation timestamp",
            data_type=DataType.DATE,
            indexing={"filterable": True, "searchable": False}
        )
    ]
}

# Cluster class schema
TENANT_CLUSTER_SCHEMA = {
    "name": "TenantCluster",
    "description": "A cluster of semantically related documents",
    "vectorizer": Configure.Vectorizer.none(),  # We'll provide our own vectors
    "multi_tenancy": Configure.multi_tenancy(
        enabled=True,
        auto_tenant_creation=True,
        auto_tenant_activation=True
    ),
    "properties": [
        Property(
            name="cluster_id",
            description="Unique identifier for the cluster",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="topic_id",
            description="ID of the parent topic",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="name",
            description="Name of the cluster",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="summary",
            description="Summary of the cluster",
            data_type=DataType.TEXT,
            indexing={"filterable": False, "searchable": True}
        ),
        Property(
            name="document_ids",
            description="List of document IDs in this cluster",
            data_type=DataType.TEXT_ARRAY,
            indexing={"filterable": True, "searchable": False}
        ),
        Property(
            name="created",
            description="Creation timestamp",
            data_type=DataType.DATE,
            indexing={"filterable": True, "searchable": False}
        )
    ]
}

# Shade class schema
TENANT_SHADE_SCHEMA = {
    "name": "TenantShade",
    "description": "A knowledge aspect extracted from document clusters",
    "vectorizer": Configure.Vectorizer.none(),  # We'll provide our own vectors
    "multi_tenancy": Configure.multi_tenancy(
        enabled=True,
        auto_tenant_creation=True,
        auto_tenant_activation=True
    ),
    "properties": [
        Property(
            name="shade_id",
            description="Unique identifier for the shade",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="name",
            description="Name of the shade",
            data_type=DataType.TEXT,
            indexing={"filterable": True, "searchable": True}
        ),
        Property(
            name="summary",
            description="Summary of the shade",
            data_type=DataType.TEXT,
            indexing={"filterable": False, "searchable": True}
        ),
        Property(
            name="content",
            description="Full content of the shade",
            data_type=DataType.TEXT,
            indexing={"filterable": False, "searchable": True}
        ),
        Property(
            name="confidence",
            description="Confidence score",
            data_type=DataType.NUMBER,
            indexing={"filterable": True, "searchable": False}
        ),
        Property(
            name="source_clusters",
            description="Source cluster IDs",
            data_type=DataType.TEXT_ARRAY,
            indexing={"filterable": True, "searchable": False}
        ),
        Property(
            name="created",
            description="Creation timestamp",
            data_type=DataType.DATE,
            indexing={"filterable": True, "searchable": False}
        ),
        Property(
            name="timeline",
            description="Timeline data for chronological events",
            data_type=DataType.TEXT,
            indexing={"filterable": False, "searchable": True}
        )
    ]
}

# All schemas
L1_SCHEMAS = [
    TENANT_TOPIC_SCHEMA,
    TENANT_CLUSTER_SCHEMA,
    TENANT_SHADE_SCHEMA
]

def get_schema_definitions():
    """Get all schema definitions for L1 in Weaviate."""
    return L1_SCHEMAS 