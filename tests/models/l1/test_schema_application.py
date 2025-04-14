import unittest
import os
from unittest.mock import patch, MagicMock

from app.models.l1.schema_application import (
    apply_postgres_schema,
    apply_weaviate_schema, 
    apply_all_schemas
)

class TestSchemaApplication(unittest.TestCase):
    """Test cases for L1 schema application."""
    
    @patch('app.models.l1.schema_application.RelationalDB')
    def test_apply_postgres_schema(self, mock_rel_db_class):
        """Test applying PostgreSQL schema."""
        # Setup mocks
        mock_rel_db = MagicMock()
        mock_rel_db_class.return_value = mock_rel_db
        
        mock_session = MagicMock()
        mock_rel_db.get_db_session.return_value = mock_session
        
        # Test successful application
        result = apply_postgres_schema(mock_rel_db)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(mock_session.execute.called)
        self.assertTrue(mock_session.commit.called)
        self.assertTrue(mock_rel_db.close_db_session.called)
    
    @patch('app.models.l1.schema_application.RelationalDB')
    def test_apply_postgres_schema_error(self, mock_rel_db_class):
        """Test applying PostgreSQL schema with error."""
        # Setup mocks
        mock_rel_db = MagicMock()
        mock_rel_db_class.return_value = mock_rel_db
        
        mock_session = MagicMock()
        mock_rel_db.get_db_session.return_value = mock_session
        
        # Make execute raise an error
        mock_session.execute.side_effect = Exception("Test error")
        
        # Test error handling
        result = apply_postgres_schema(mock_rel_db)
        
        # Verify
        self.assertFalse(result)
        self.assertTrue(mock_session.execute.called)
        self.assertTrue(mock_session.rollback.called)
        self.assertTrue(mock_rel_db.close_db_session.called)
    
    @patch('app.models.l1.schema_application.VectorDB')
    def test_apply_weaviate_schema(self, mock_vector_db_class):
        """Test applying Weaviate schema."""
        # Setup mocks
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db
        
        # Mock collections
        mock_collections = MagicMock()
        mock_vector_db.client.collections = mock_collections
        mock_collections.list_all.return_value = []
        
        # Test successful application
        result = apply_weaviate_schema(mock_vector_db)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(mock_collections.list_all.called)
        self.assertTrue(mock_collections.create.called)
    
    @patch('app.models.l1.schema_application.VectorDB')
    def test_apply_weaviate_schema_existing_collections(self, mock_vector_db_class):
        """Test applying Weaviate schema with existing collections."""
        # Setup mocks
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db
        
        # Mock collections with existing collections
        mock_collections = MagicMock()
        mock_vector_db.client.collections = mock_collections
        mock_collections.list_all.return_value = ["TenantTopic", "TenantCluster", "TenantShade"]
        
        # Test application with existing collections
        result = apply_weaviate_schema(mock_vector_db)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(mock_collections.list_all.called)
        # Create should not be called for existing collections
        self.assertFalse(mock_collections.create.called)
    
    @patch('app.models.l1.schema_application.apply_postgres_schema')
    @patch('app.models.l1.schema_application.apply_weaviate_schema')
    def test_apply_all_schemas(self, mock_apply_weaviate, mock_apply_postgres):
        """Test applying all schemas."""
        # Setup mocks to return success
        mock_apply_postgres.return_value = True
        mock_apply_weaviate.return_value = True
        
        # Test successful application
        result = apply_all_schemas()
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(mock_apply_postgres.called)
        self.assertTrue(mock_apply_weaviate.called)
    
    @patch('app.models.l1.schema_application.apply_postgres_schema')
    @patch('app.models.l1.schema_application.apply_weaviate_schema')
    def test_apply_all_schemas_partial_failure(self, mock_apply_weaviate, mock_apply_postgres):
        """Test applying all schemas with partial failure."""
        # Setup mocks with one failure
        mock_apply_postgres.return_value = True
        mock_apply_weaviate.return_value = False
        
        # Test partial failure
        result = apply_all_schemas()
        
        # Verify
        self.assertFalse(result)
        self.assertTrue(mock_apply_postgres.called)
        self.assertTrue(mock_apply_weaviate.called)

if __name__ == "__main__":
    unittest.main() 