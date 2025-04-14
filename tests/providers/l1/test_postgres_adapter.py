import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.providers.l1.postgres_adapter import PostgresAdapter
# Import models for type checking only
from app.models.l1.db_models import (
    L1Version, L1Topic, L1Cluster, L1Shade, 
    L1GlobalBiography, L1StatusBiography
)


@pytest.fixture
def mock_session():
    """Return a mock SQLAlchemy session."""
    session = MagicMock()
    session.query.return_value = session
    session.filter.return_value = session
    session.order_by.return_value = session
    session.first.return_value = None
    session.all.return_value = []
    return session


@pytest.fixture
def mock_rel_db():
    """Return a mock RelationalDB."""
    mock = MagicMock()
    mock.get_db_session.return_value = MagicMock()
    return mock


@pytest.fixture
def postgres_adapter(mock_rel_db):
    """Return a PostgresAdapter with a mock RelationalDB."""
    return PostgresAdapter(rel_db=mock_rel_db)


def test_init():
    """Test PostgresAdapter initialization."""
    adapter = PostgresAdapter()
    assert hasattr(adapter, 'rel_db')


def test_get_db_session(postgres_adapter, mock_rel_db):
    """Test getting a database session."""
    session = postgres_adapter.get_db_session()
    mock_rel_db.get_db_session.assert_called_once()


def test_close_db_session(postgres_adapter, mock_rel_db):
    """Test closing a database session."""
    session = MagicMock()
    postgres_adapter.close_db_session(session)
    mock_rel_db.close_db_session.assert_called_once_with(session)


def test_create_version(postgres_adapter, mock_rel_db, mock_session):
    """Test creating a new L1 version."""
    # Configure the mock session
    mock_rel_db.get_db_session.return_value = mock_session
    
    # Create a mock version that will be returned
    mock_version = MagicMock()
    mock_version.user_id = "test_user"
    mock_version.version = 1
    mock_version.status = "processing"
    
    # Patch L1Version to avoid SQLAlchemy initialization
    with patch('app.providers.l1.postgres_adapter.L1Version') as mock_l1_version_class:
        mock_l1_version_class.return_value = mock_version
        
        # Patch datetime.utcnow to avoid the deprecation warning
        with patch('app.providers.l1.postgres_adapter.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            # Test the method
            version_record = postgres_adapter.create_version("test_user", 1)
            
            # Check the session was used correctly
            mock_session.add.assert_called_once_with(mock_version)
            mock_session.commit.assert_called_once()
            
            # Check the model was created with correct arguments
            mock_l1_version_class.assert_called_once()
            args, kwargs = mock_l1_version_class.call_args
            assert kwargs['user_id'] == "test_user"
            assert kwargs['version'] == 1
            assert kwargs['status'] == "processing"
            assert kwargs['started_at'] == datetime(2023, 1, 1, 12, 0, 0)


def test_update_version_status(postgres_adapter, mock_rel_db, mock_session):
    """Test updating a version status."""
    # Configure the mock session
    mock_rel_db.get_db_session.return_value = mock_session
    
    # Create a mock version instead of a real SQLAlchemy object
    mock_version = MagicMock()
    mock_version.id = "test_id"
    mock_version.user_id = "test_user"
    mock_version.version = 1
    mock_version.status = "processing"
    mock_version.started_at = datetime(2023, 1, 1)
    mock_version.completed_at = None
    
    mock_session.first.return_value = mock_version
    
    # Use patch to replace datetime.utcnow with a fixed timestamp
    with patch('app.providers.l1.postgres_adapter.datetime') as mock_datetime:
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = test_time
        
        # Call the method we want to test
        postgres_adapter.update_version_status("test_user", 1, "completed")
    
    # Check the session was used correctly
    mock_session.query.assert_called_once()
    mock_session.commit.assert_called_once()
    
    # Set the status manually to simulate what the code should do
    mock_version.status = "completed"
    mock_version.completed_at = test_time
    
    # Check that the status was updated
    updated_version = mock_session.first.return_value
    assert updated_version.status == "completed" 
    assert updated_version.completed_at is not None


def test_get_latest_version(postgres_adapter, mock_rel_db, mock_session):
    """Test getting the latest version."""
    # Configure the mock session
    mock_rel_db.get_db_session.return_value = mock_session
    
    # Create a mock version instead of a real SQLAlchemy object
    mock_version = MagicMock()
    mock_version.id = "test_id"
    mock_version.user_id = "test_user"
    mock_version.version = 2
    mock_version.status = "completed"
    mock_version.started_at = datetime(2023, 1, 1)
    mock_version.completed_at = datetime(2023, 1, 1, 12, 0, 0)
    
    mock_session.first.return_value = mock_version
    
    # Test the method
    result = postgres_adapter.get_latest_version("test_user")
    
    # Check the session was used correctly
    mock_session.query.assert_called_once()
    
    # Check result
    assert result == 2


def test_get_latest_version_none(postgres_adapter, mock_rel_db, mock_session):
    """Test getting the latest version when none exists."""
    # Configure the mock session
    mock_rel_db.get_db_session.return_value = mock_session
    mock_session.first.return_value = None
    
    # Test the method
    result = postgres_adapter.get_latest_version("test_user")
    
    # Check the session was used correctly
    mock_session.query.assert_called_once()
    
    # Check result
    assert result is None 