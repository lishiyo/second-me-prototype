import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Union

import weaviate
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.tenants import Tenant, TenantActivityStatus

logger = logging.getLogger(__name__)

class VectorDB:
    """
    A class for interacting with Weaviate v4 vector database.
    This adapter provides methods to store, search, and manage vector embeddings.
    """
    
    DEFAULT_CLASS_NAME = "TenantChunk"
    DOCUMENT_CLASS_NAME = "Document"
    
    def __init__(self, 
                 url: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        """
        Initialize the Weaviate client.
        
        Args:
            url: Weaviate cluster URL (defaults to env var WEAVIATE_URL)
            api_key: Weaviate API key (defaults to env var WEAVIATE_API_KEY)
            embedding_model: Embedding model to use (defaults to env var EMBEDDING_MODEL)
        """
        self.url = url or os.environ.get('WEAVIATE_URL')
        self.api_key = api_key or os.environ.get('WEAVIATE_API_KEY')
        self.embedding_model = embedding_model or os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
        
        if not self.url or not self.api_key:
            raise ValueError("Missing required Weaviate configuration")
        
        # Create Weaviate client using connect_to_weaviate_cloud helper
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key),
            headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY', '')
            }
        )
        
        # Ensure the class schemas exist
        self._ensure_schemas_exist()
    
    def _ensure_schemas_exist(self) -> None:
        """
        Ensure the required Weaviate class schemas exist, create them if they don't.
        """
        try:
            # Check which collections exist
            collections = self.client.collections.list_all()
            collection_names = collections  # collections is already a list of names in v4
            
            # Create TenantChunk schema if needed
            if self.DEFAULT_CLASS_NAME not in collection_names:
                self._create_chunk_schema()
                
            # Create Document schema if needed
            if self.DOCUMENT_CLASS_NAME not in collection_names:
                self._create_document_schema()
                
        except Exception as e:
            logger.error(f"Error creating schemas: {e}")
            raise
            
    def _create_chunk_schema(self) -> None:
        """Create the TenantChunk schema for storing chunk-level embeddings."""
        logger.info(f"Creating collection schema {self.DEFAULT_CLASS_NAME}")
        
        # Create the collection with proper configuration
        self.client.collections.create(
            name=self.DEFAULT_CLASS_NAME,
            description="A chunk of text from a document with tenant isolation",
            # Use pre-computed embeddings from OpenAI rather than Weaviate's vectorizer
            vectorizer_config=Configure.Vectorizer.none(),
            # Configure multi-tenancy
            multi_tenancy_config=Configure.multi_tenancy(
                enabled=True,
                auto_tenant_creation=True  # Automatically create tenants when needed
            ),
            # Define properties
            properties=[
                Property(
                    name="document_id",
                    description="Source document ID",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="s3_path",
                    description="Path to the chunk file in Wasabi S3",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="filename",
                    description="Original filename",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="content_type",
                    description="Content type of the document",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="timestamp",
                    description="Timestamp of chunk creation",
                    data_type=DataType.DATE,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="chunk_index",
                    description="Index of chunk in original document",
                    data_type=DataType.INT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="topic",
                    description="Topic classification for the chunk",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="tags",
                    description="List of tags associated with the chunk",
                    data_type=DataType.TEXT_ARRAY,
                    indexing={"filterable": True, "searchable": True},
                ),
            ]
        )
        
    def _create_document_schema(self) -> None:
        """Create the Document schema for storing document-level embeddings."""
        logger.info(f"Creating collection schema {self.DOCUMENT_CLASS_NAME}")
        
        # Create the document collection with proper configuration
        self.client.collections.create(
            name=self.DOCUMENT_CLASS_NAME,
            description="Document-level embedding and metadata with tenant isolation",
            # Use pre-computed embeddings from OpenAI rather than Weaviate's vectorizer
            vectorizer_config=Configure.Vectorizer.none(),
            # Configure multi-tenancy
            multi_tenancy_config=Configure.multi_tenancy(
                enabled=True,
                auto_tenant_creation=True  # Automatically create tenants when needed
            ),
            # Define properties
            properties=[
                Property(
                    name="document_id",
                    description="Source document ID",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="s3_path",
                    description="Path to the document in Wasabi S3",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="title",
                    description="Document title",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="filename",
                    description="Original filename",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="content_type",
                    description="Content type of the document",
                    data_type=DataType.TEXT,
                    indexing={"filterable": True, "searchable": True},
                ),
                Property(
                    name="timestamp",
                    description="Timestamp of document creation",
                    data_type=DataType.DATE,
                    indexing={"filterable": True, "searchable": True},
                ),
            ]
        )
    
    @staticmethod
    def generate_consistent_id(tenant_id: str, document_id: str, chunk_index: int) -> str:
        """
        Generate a consistent UUID for a chunk based on tenant, document, and chunk index.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID
            chunk_index: Chunk index
            
        Returns:
            UUID string
        """
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace
        name = f"{tenant_id}:{document_id}:{chunk_index}"
        return str(uuid.uuid5(namespace, name))
    
    def get_collection(self):
        """Get the collection object for the default class"""
        return self.client.collections.get(self.DEFAULT_CLASS_NAME)
    
    def get_tenant_collection(self, tenant_id: str):
        """Get a tenant-specific collection object"""
        collection = self.get_collection()
        return collection.with_tenant(tenant=tenant_id)
    
    def add_chunk(self, 
                 tenant_id: str,
                 document_id: str,
                 s3_path: str,
                 chunk_index: int, 
                 embedding: List[float],
                 metadata: Dict[str, Any]) -> str:
        """
        Add a document chunk to the vector store.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Source document ID
            s3_path: Path to the chunk file in Wasabi S3
            chunk_index: Index of chunk in original document
            embedding: Pre-computed embedding vector for the chunk
            metadata: Additional metadata
            
        Returns:
            UUID of the added object
        """
        try:
            obj_id = VectorDB.generate_consistent_id(tenant_id, document_id, chunk_index)
            
            # Prepare the object data
            properties = {
                "document_id": document_id,
                "s3_path": s3_path,
                "chunk_index": chunk_index,
                "filename": metadata.get("filename", ""),
                "content_type": metadata.get("content_type", ""),
                "timestamp": metadata.get("timestamp", ""),
                "topic": metadata.get("topic", ""),  # Add topic from metadata
                "tags": metadata.get("tags", [])     # Add tags from metadata
            }
            
            # Get the tenant-specific collection
            tenant_collection = self.get_tenant_collection(tenant_id)
            
            # Add the object using the tenant-specific collection with pre-computed embedding
            result = tenant_collection.data.insert(
                properties=properties,
                uuid=obj_id,
                vector=embedding
            )
                
            return obj_id
        except Exception as e:
            logger.error(f"Error adding chunk for document {document_id}: {e}")
            raise
    
    def batch_add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple document chunks in a batch operation.
        
        Args:
            chunks: List of chunk dictionaries with the following keys:
                - tenant_id: Tenant/user ID
                - document_id: Source document ID
                - s3_path: Path to the chunk file in Wasabi S3
                - chunk_index: Index of chunk in original document
                - embedding: Pre-computed embedding vector for the chunk
                - metadata: Additional metadata dictionary
                
        Returns:
            List of UUIDs for the added objects
        """
        try:
            uuids = []
            
            # Group chunks by tenant for efficient batch processing
            tenant_chunks = {}
            for chunk in chunks:
                tenant_id = chunk["tenant_id"]
                if tenant_id not in tenant_chunks:
                    tenant_chunks[tenant_id] = []
                tenant_chunks[tenant_id].append(chunk)
            
            # Process each tenant's chunks separately
            for tenant_id, tenant_chunk_list in tenant_chunks.items():
                tenant_collection = self.get_tenant_collection(tenant_id)
                
                # Process the chunks
                with tenant_collection.batch.dynamic() as batch:
                    for chunk in tenant_chunk_list:
                        obj_id = VectorDB.generate_consistent_id(
                            chunk["tenant_id"], 
                            chunk["document_id"], 
                            chunk["chunk_index"]
                        )
                        uuids.append(obj_id)
                        
                        # Format timestamp to RFC3339 if needed
                        timestamp = chunk["metadata"].get("timestamp", "")
                        if timestamp and "." in timestamp:
                            # Remove microseconds if present to ensure RFC3339 compliance
                            timestamp = timestamp.split(".")[0] + "Z"
                        
                        # Prepare the object data
                        properties = {
                            "document_id": chunk["document_id"],
                            "s3_path": chunk["s3_path"],
                            "chunk_index": chunk["chunk_index"],
                            "filename": chunk["metadata"].get("filename", ""),
                            "content_type": chunk["metadata"].get("content_type", ""),
                            "timestamp": timestamp,
                            "topic": chunk["metadata"].get("topic", ""),  # Add topic from metadata
                            "tags": chunk["metadata"].get("tags", [])     # Add tags from metadata
                        }
                        
                        # Add to batch with embedding vector
                        batch.add_object(
                            properties=properties,
                            uuid=obj_id,
                            vector=chunk.get("embedding")  # Add the pre-computed embedding
                        )
                    
            return uuids
        except Exception as e:
            logger.error(f"Error batch adding chunks: {e}")
            raise
    
    def search(self, 
              tenant_id: str,
              query: str, 
              limit: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for chunks semantically similar to the query.
        
        Args:
            tenant_id: Tenant/user ID to search in
            query: Search query text
            limit: Maximum number of results to return
            filters: Additional filters to apply
            
        Returns:
            List of matching objects with their properties and scores
        """
        try:
            # Get the tenant-specific collection
            tenant_collection = self.get_tenant_collection(tenant_id)
            
            # Build the filter if provided
            filter_query = None
            if filters:
                filter_query = Filter.all_of()
                for prop, value in filters.items():
                    if isinstance(value, list):
                        filter_query = filter_query.add(
                            Filter.by_property(prop).contains_any(value)
                        )
                    else:
                        filter_query = filter_query.add(
                            Filter.by_property(prop).equal(value)
                        )
            
            # Execute the semantic search
            results = tenant_collection.query.near_text(
                query=query,
                limit=limit,
                filters=filter_query,
                return_properties=["document_id", "s3_path", "chunk_index", 
                                  "filename", "content_type", "timestamp"],
                return_metadata=["distance"]
            )
            
            # Process results
            if not results.objects:
                return []
                
            objects = []
            for obj in results.objects:
                properties = obj.properties
                properties["score"] = 1.0 - obj.metadata.distance
                objects.append(properties)
                    
            return objects
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            raise
    
    def hybrid_search(self, 
                     tenant_id: str,
                     query: str, 
                     limit: int = 10, 
                     filters: Optional[Dict[str, Any]] = None,
                     alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both vector and keyword search.
        
        Args:
            tenant_id: Tenant/user ID to search in
            query: Search query text
            limit: Maximum number of results to return
            filters: Additional filters to apply
            alpha: Weight between vector (0) and keyword (1) search
            
        Returns:
            List of matching objects with their properties and scores
        """
        try:
            # Get the tenant-specific collection
            tenant_collection = self.get_tenant_collection(tenant_id)
            
            # Build the filter if provided
            filter_query = None
            if filters:
                filter_query = Filter.all_of()
                for prop, value in filters.items():
                    if isinstance(value, list):
                        filter_query = filter_query.add(
                            Filter.by_property(prop).contains_any(value)
                        )
                    else:
                        filter_query = filter_query.add(
                            Filter.by_property(prop).equal(value)
                        )
            
            # Execute the hybrid search
            results = tenant_collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                filters=filter_query,
                return_properties=["document_id", "s3_path", "chunk_index", 
                                  "filename", "content_type", "timestamp"],
                return_metadata=["score"]
            )
            
            # Process results
            if not results.objects:
                return []
                
            objects = []
            for obj in results.objects:
                properties = obj.properties
                properties["score"] = obj.metadata.score
                objects.append(properties)
                    
            return objects
        except Exception as e:
            logger.error(f"Error performing hybrid search for '{query}': {e}")
            raise
    
    def get_by_id(self, tenant_id: str, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get an object by its UUID.
        
        Args:
            tenant_id: Tenant/user ID
            uuid: UUID of the object
            
        Returns:
            Object properties if found, None otherwise
        """
        try:
            # Get the tenant-specific collection
            tenant_collection = self.get_tenant_collection(tenant_id)
            
            # Get the object
            result = tenant_collection.query.fetch_object_by_id(
                uuid=uuid,
                return_properties=["document_id", "s3_path", "chunk_index", 
                                  "filename", "content_type", "timestamp"]
            )
            
            if not result:
                return None
                
            return result.properties
        except Exception as e:
            logger.error(f"Error getting object by ID {uuid}: {e}")
            raise
    
    def delete_by_id(self, tenant_id: str, uuid: str) -> bool:
        """
        Delete an object by its UUID.
        
        Args:
            tenant_id: Tenant/user ID
            uuid: UUID of the object
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            # Get the tenant-specific collection
            tenant_collection = self.get_tenant_collection(tenant_id)
            
            # Delete the object
            tenant_collection.data.delete_by_id(uuid=uuid)
            return True
        except Exception as e:
            logger.error(f"Error deleting object by ID {uuid}: {e}")
            return False
    
    def delete_by_document(self, tenant_id: str, document_id: str) -> int:
        """
        Delete all chunks associated with a document.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID
            
        Returns:
            Number of objects deleted
        """
        try:
            # Get the tenant-specific collection
            tenant_collection = self.get_tenant_collection(tenant_id)
            
            # Find all chunks for this document
            filter_query = Filter.by_property("document_id").equal(document_id)
            
            # In v4, we need to specify return_metadata differently
            results = tenant_collection.query.fetch_objects(
                filters=filter_query,
                return_properties=["document_id"],
            )
            
            if not results.objects:
                return 0
                
            # Delete each object
            count = 0
            for obj in results.objects:
                if tenant_collection.data.delete_by_id(uuid=obj.uuid):
                    count += 1
                    
            return count
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {e}")
            raise
    
    def list_tenants(self) -> List[str]:
        """
        List all tenants in the collection.
        
        Returns:
            List of tenant IDs
        """
        try:
            collection = self.get_collection()
            # In v4, tenants.get() directly returns a list of tenant names
            return collection.tenants.get()
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            raise
            
    def create_tenant(self, tenant_id: str) -> bool:
        """
        Create a new tenant.
        
        Args:
            tenant_id: Tenant/user ID
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            collection = self.get_collection()
            collection.tenants.create(tenant_id)
            return True
        except Exception as e:
            logger.error(f"Error creating tenant {tenant_id}: {e}")
            return False
            
    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete a tenant and all its objects.
        
        Args:
            tenant_id: Tenant/user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_collection()
            # Get tenants dictionary
            tenants = collection.tenants.get()
            if tenant_id in tenants:
                collection.tenants.remove(tenant_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting tenant {tenant_id}: {e}")
            return False
            
    def get_tenant_status(self, tenant_id: str) -> str:
        """
        Get the status of a tenant.
        
        Args:
            tenant_id: Tenant/user ID
            
        Returns:
            Status of the tenant ('ACTIVE' or 'INACTIVE')
        """
        try:
            collection = self.get_collection()
            tenants = collection.tenants.get()
            
            if tenant_id in tenants:
                tenant_info = tenants[tenant_id]
                # In v1.26+, ACTIVE replaces HOT, and INACTIVE replaces COLD
                return tenant_info.activityStatus.value
            return "NOT_FOUND"
        except Exception as e:
            logger.error(f"Error getting tenant status for {tenant_id}: {e}")
            raise
            
    def set_tenant_status(self, tenant_id: str, active: bool) -> bool:
        """
        Set the status of a tenant.
        
        Args:
            tenant_id: Tenant/user ID
            active: Whether the tenant should be active
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_collection()
            tenants = collection.tenants.get()
            
            if tenant_id in tenants:
                # Get the tenant info to update it
                tenant_info = tenants[tenant_id]
                status = TenantActivityStatus.ACTIVE if active else TenantActivityStatus.INACTIVE
                
                # In v4, we need to use the update method with the tenant name
                # Documentation shows using tenant.update() directly on tenant object
                collection.tenants.update(tenants=[
                    Tenant(
                        name=tenant_info.name,
                        activity_status=status # INACTIVE, ACTIVE
                    )
                ])
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting tenant status for {tenant_id}: {e}")
            return False
            
    def close(self) -> None:
        """
        Close the Weaviate client connection and release resources.
        Should be called when done using the VectorDB instance.
        """
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.debug("Weaviate client connection closed")

    def get_document_collection(self):
        """Get the collection object for the document class"""
        return self.client.collections.get(self.DOCUMENT_CLASS_NAME)
    
    def get_tenant_document_collection(self, tenant_id: str):
        """Get a tenant-specific document collection object"""
        collection = self.get_document_collection()
        return collection.with_tenant(tenant=tenant_id)
    
    def add_document(self, 
                   tenant_id: str,
                   document_id: str,
                   s3_path: str,
                   embedding: List[float],
                   metadata: Dict[str, Any]) -> str:
        """
        Add a document-level embedding to the vector store.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Source document ID
            s3_path: Path to the document in Wasabi S3
            embedding: Pre-computed embedding vector for the entire document
            metadata: Additional metadata including title, filename, content_type, etc.
            
        Returns:
            UUID of the added object
        """
        try:
            # Generate a consistent UUID for this document
            obj_id = self.generate_consistent_document_id(tenant_id, document_id)
            
            # Prepare the object data
            properties = {
                "document_id": document_id,
                "s3_path": s3_path,
                "title": metadata.get("title", ""),
                "filename": metadata.get("filename", ""),
                "content_type": metadata.get("content_type", ""),
                "timestamp": metadata.get("timestamp", "")
            }
            
            # Get the tenant-specific document collection
            tenant_collection = self.get_tenant_document_collection(tenant_id)
            
            # Add the object using the tenant-specific collection with pre-computed embedding
            tenant_collection.data.insert(
                properties=properties,
                uuid=obj_id,
                vector=embedding
            )
                
            logger.info(f"Added document {document_id} to vector DB with UUID {obj_id}")
            return obj_id
        except Exception as e:
            logger.error(f"Error adding document {document_id}: {e}")
            raise
    
    def generate_consistent_document_id(self, tenant_id: str, document_id: str) -> str:
        """
        Generate a consistent UUID for a document based on tenant and document ID.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID
            
        Returns:
            UUID string
        """
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace
        name = f"{tenant_id}:{document_id}:document"
        return str(uuid.uuid5(namespace, name))
    
    def get_document_embedding(self, tenant_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document-level embedding from the vector store.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID
            
        Returns:
            Dictionary with document properties and embedding, or None if not found
        """
        try:
            # Generate the consistent document UUID
            obj_id = self.generate_consistent_document_id(tenant_id, document_id)
            
            # Get the tenant-specific document collection
            tenant_collection = self.get_tenant_document_collection(tenant_id)
            
            # Get the object with its vector
            response = tenant_collection.query.fetch_object_by_id(
                uuid=obj_id,
                include_vector=True
            )
            
            if response:
                # Extract the properties and vector
                return {
                    "document_id": document_id,
                    "properties": response.properties,
                    "embedding": response.vector
                }
            
            return None
        except Exception as e:
            logger.error(f"Error getting document embedding for {document_id}: {e}")
            return None
    
    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        """
        Delete a document-level embedding from the vector store.
        
        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate the consistent document UUID
            obj_id = self.generate_consistent_document_id(tenant_id, document_id)
            
            # Get the tenant-specific document collection
            tenant_collection = self.get_tenant_document_collection(tenant_id)
            
            # Delete the object
            tenant_collection.data.delete_by_id(uuid=obj_id)
            logger.info(f"Deleted document {document_id} from vector DB")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False 