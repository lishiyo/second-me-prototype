import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.providers.l1.postgres_adapter import PostgresAdapter
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
    
    # Test the method
    version_record = postgres_adapter.create_version("test_user", 1)
    
    # Check the session was used correctly
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    
    # Check that the added object was a L1Version
    added_obj = mock_session.add.call_args[0][0]
    assert isinstance(added_obj, L1Version)
    assert added_obj.user_id == "test_user"
    assert added_obj.version == 1
    assert added_obj.status == "processing"


def test_update_version_status(postgres_adapter, mock_rel_db, mock_session):
    """Test updating a version status."""
    # Configure the mock session
    mock_rel_db.get_db_session.return_value = mock_session
    mock_session.first.return_value = L1Version(
        id="test_id",
        user_id="test_user",
        version=1,
        status="processing",
        started_at=datetime.utcnow()
    )
    
    # Test the method
    postgres_adapter.update_version_status("test_user", 1, "completed")
    
    # Check the session was used correctly
    mock_session.query.assert_called_once()
    mock_session.commit.assert_called_once()
    
    # Check that the status was updated
    updated_version = mock_session.first.return_value
    assert updated_version.status == "completed"
    assert updated_version.completed_at is not None


def test_get_latest_version(postgres_adapter, mock_rel_db, mock_session):
    """Test getting the latest version."""
    # Configure the mock session
    mock_rel_db.get_db_session.return_value = mock_session
    mock_session.first.return_value = L1Version(
        id="test_id",
        user_id="test_user",
        version=2,
        status="completed",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )
    
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