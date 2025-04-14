import unittest
import os
import uuid
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError

from app.providers.rel_db import RelationalDB, User, Document, TrainingJob, ChatSession


class TestRelationalDB(unittest.TestCase):
    """Test cases for the RelationalDB class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock engine and session
        self.mock_engine = MagicMock()
        self.mock_session = MagicMock()
        self.mock_session_local = MagicMock()
        self.mock_session_local.return_value = self.mock_session

        # Patch SQLAlchemy engine and sessionmaker
        self.engine_patch = patch('app.providers.rel_db.create_engine', return_value=self.mock_engine)
        self.sessionmaker_patch = patch('app.providers.rel_db.sessionmaker', return_value=self.mock_session_local)
        self.create_tables_patch = patch.object(RelationalDB, '_create_tables')

        # Start patches
        self.engine_patch.start()
        self.sessionmaker_patch.start()
        self.create_tables_patch.start()

        # Create test instance with mock env vars
        self.test_db = RelationalDB(
            host="localhost",
            port="5432",
            database="test_db",
            user="test_user",
            password="test_password"
        )

    def tearDown(self):
        """Clean up test environment after each test."""
        # Stop patches
        self.engine_patch.stop()
        self.sessionmaker_patch.stop()
        self.create_tables_patch.stop()

    def test_init_with_params(self):
        """Test initialization with parameters."""
        db = RelationalDB(
            host="test.host",
            port="1234",
            database="db_name",
            user="username",
            password="password"
        )
        
        expected_conn_string = "postgresql://username:password@test.host:1234/db_name"
        self.assertEqual(db.connection_string, expected_conn_string)

    def test_init_with_connection_string(self):
        """Test initialization with direct connection string."""
        conn_string = "postgresql://user:pass@host:5432/db"
        db = RelationalDB(connection_string=conn_string)
        
        self.assertEqual(db.connection_string, conn_string)

    def test_init_missing_params(self):
        """Test initialization with missing parameters."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                RelationalDB()

    def test_get_db_session(self):
        """Test getting a database session."""
        session = self.test_db.get_db_session()
        
        self.assertEqual(session, self.mock_session)
        self.mock_session_local.assert_called_once()

    def test_close_db_session(self):
        """Test closing a database session."""
        self.test_db.close_db_session(self.mock_session)
        
        self.mock_session.close.assert_called_once()

    # User operations tests
    def test_create_user(self):
        """Test creating a user."""
        # Create a mock user instance that will be returned
        mock_user = MagicMock(spec=User)
        mock_user.id = "test-user-id"
        
        # Set up the mock to intercept User instantiation
        with patch('app.providers.rel_db.User') as mock_user_class:
            mock_user_class.return_value = mock_user
            
            # Configure commit to do nothing
            self.mock_session.commit.return_value = None
            
            # Call the method to test
            result = self.test_db.create_user(self.mock_session, user_id="test-user-id")
            
            # Verify results
            self.mock_session.add.assert_called_once()
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_user)
            # Verify the User was instantiated with correct ID
            mock_user_class.assert_called_once_with(id="test-user-id")

    def test_create_user_error(self):
        """Test error handling during user creation."""
        # Make session.add raise an exception
        self.mock_session.add.side_effect = SQLAlchemyError("Test error")
        
        # Call the method and check exception handling
        with self.assertRaises(SQLAlchemyError):
            self.test_db.create_user(self.mock_session)
        
        # Verify rollback was called
        self.mock_session.rollback.assert_called_once()

    def test_get_user(self):
        """Test getting a user by ID."""
        # Create a mock user to be returned by the query
        mock_user = MagicMock(spec=User)
        mock_user.id = "test-user-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_user
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_user(self.mock_session, "test-user-id")
        
        # Verify results
        self.assertEqual(result, mock_user)
        self.mock_session.query.assert_called_once_with(User)

    def test_get_or_create_user_existing(self):
        """Test getting an existing user."""
        # Create a mock user to be returned by the query
        mock_user = MagicMock(spec=User)
        mock_user.id = "test-user-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_user
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_or_create_user(self.mock_session, "test-user-id")
        
        # Verify results
        self.assertEqual(result, mock_user)
        self.mock_session.add.assert_not_called()  # Shouldn't create when found

    def test_get_or_create_user_new(self):
        """Test creating a user when not found."""
        # Set up query to return None (user not found)
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        self.mock_session.query.return_value = mock_query
        
        # Create a mock user for the result
        mock_user = MagicMock(spec=User)
        mock_user.id = "test-user-id"
        
        # Set up the mock to intercept User instantiation
        with patch('app.providers.rel_db.User') as mock_user_class:
            mock_user_class.return_value = mock_user
            
            # Call the method to test
            result = self.test_db.get_or_create_user(self.mock_session, "test-user-id")
            
            # Verify results
            self.mock_session.add.assert_called_once()
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_user)
            # Verify the User was instantiated with correct ID
            mock_user_class.assert_called_once_with(id="test-user-id")

    # Document operations tests
    def test_create_document(self):
        """Test creating a document."""
        # Create mock document
        mock_document = MagicMock(spec=Document)
        mock_document.id = "test-doc-id"
        
        # Set up the mock to intercept Document instantiation
        with patch('app.providers.rel_db.Document') as mock_document_class:
            mock_document_class.return_value = mock_document
            
            # Call the method to test
            result = self.test_db.create_document(
                self.mock_session,
                user_id="test-user-id",
                filename="test.pdf",
                content_type="application/pdf",
                s3_path="test/path.pdf"
            )
            
            # Verify results
            self.mock_session.add.assert_called_once()
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_document)
            # Verify Document was instantiated with correct params
            mock_document_class.assert_called_once_with(
                user_id="test-user-id",
                filename="test.pdf",
                content_type="application/pdf",
                s3_path="test/path.pdf"
            )

    def test_get_document(self):
        """Test getting a document by ID."""
        # Create a mock document
        mock_document = MagicMock(spec=Document)
        mock_document.id = "test-doc-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_document
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_document(self.mock_session, "test-doc-id")
        
        # Verify results
        self.assertEqual(result, mock_document)
        self.mock_session.query.assert_called_once_with(Document)

    def test_get_user_documents(self):
        """Test getting all documents for a user."""
        # Create mock documents list
        mock_doc1 = MagicMock(spec=Document)
        mock_doc1.id = "test-doc-1"
        mock_doc2 = MagicMock(spec=Document)
        mock_doc2.id = "test-doc-2"
        mock_documents = [mock_doc1, mock_doc2]
        
        # Set up the mock for session.query().filter().all()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_documents
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_user_documents(self.mock_session, "test-user-id")
        
        # Verify results
        self.assertEqual(result, mock_documents)
        self.mock_session.query.assert_called_once_with(Document)

    def test_update_document_processed(self):
        """Test updating document processed status."""
        # Create a mock document
        mock_document = MagicMock(spec=Document)
        mock_document.id = "test-doc-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_document
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.update_document_processed(
            self.mock_session,
            "test-doc-id",
            True,
            10
        )
        
        # Verify results
        self.assertEqual(result, mock_document)
        self.assertEqual(mock_document.processed, True)
        self.assertEqual(mock_document.chunk_count, 10)
        self.mock_session.commit.assert_called_once()

    def test_delete_document(self):
        """Test deleting a document."""
        # Create a mock document
        mock_document = MagicMock(spec=Document)
        mock_document.id = "test-doc-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_document
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.delete_document(self.mock_session, "test-doc-id")
        
        # Verify results
        self.assertTrue(result)
        self.mock_session.delete.assert_called_once_with(mock_document)
        self.mock_session.commit.assert_called_once()

    # Training job operations tests
    def test_create_training_job(self):
        """Test creating a training job."""
        # Create mock training job
        mock_job = MagicMock(spec=TrainingJob)
        mock_job.id = "test-job-id"
        mock_job.status = "queued"
        
        # Set up the mock to intercept TrainingJob instantiation
        with patch('app.providers.rel_db.TrainingJob') as mock_training_job_class:
            mock_training_job_class.return_value = mock_job
            
            # Call the method to test
            result = self.test_db.create_training_job(self.mock_session, "test-user-id")
            
            # Verify results
            self.mock_session.add.assert_called_once()
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_job)
            # Verify TrainingJob was instantiated with correct params
            mock_training_job_class.assert_called_once()
            args, kwargs = mock_training_job_class.call_args
            self.assertEqual(kwargs["user_id"], "test-user-id")
            self.assertEqual(kwargs["status"], "queued")
        
    def test_get_training_job(self):
        """Test getting a training job by ID."""
        # Create a mock training job
        mock_job = MagicMock(spec=TrainingJob)
        mock_job.id = "test-job-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_job
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_training_job(self.mock_session, "test-job-id")
        
        # Verify results
        self.assertEqual(result, mock_job)
        self.mock_session.query.assert_called_once_with(TrainingJob)
        
    def test_get_user_training_jobs(self):
        """Test getting all training jobs for a user."""
        # Create mock jobs list
        mock_job1 = MagicMock(spec=TrainingJob)
        mock_job1.id = "test-job-1"
        mock_job2 = MagicMock(spec=TrainingJob)
        mock_job2.id = "test-job-2"
        mock_jobs = [mock_job1, mock_job2]
        
        # Set up the mock for session.query().filter().all()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_jobs
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_user_training_jobs(self.mock_session, "test-user-id")
        
        # Verify results
        self.assertEqual(result, mock_jobs)
        self.mock_session.query.assert_called_once_with(TrainingJob)
        
    def test_update_training_job_status(self):
        """Test updating training job status."""
        # Create a mock training job
        mock_job = MagicMock(spec=TrainingJob)
        mock_job.id = "test-job-id"
        mock_job.status = "queued"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_job
        self.mock_session.query.return_value = mock_query
        
        # Current time for testing
        test_time = datetime.now(timezone.utc)
        
        # Call the method to test with additional fields
        result = self.test_db.update_training_job_status(
            self.mock_session,
            "test-job-id",
            "processing",
            started_at=test_time,
            error="No error"
        )
        
        # Verify results
        self.assertEqual(result, mock_job)
        self.assertEqual(mock_job.status, "processing")
        self.assertEqual(mock_job.started_at, test_time)
        self.assertEqual(mock_job.error, "No error")
        self.mock_session.commit.assert_called_once()
        
    def test_delete_training_job(self):
        """Test deleting a training job."""
        # Create a mock training job
        mock_job = MagicMock(spec=TrainingJob)
        mock_job.id = "test-job-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_job
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.delete_training_job(self.mock_session, "test-job-id")
        
        # Verify results
        self.assertTrue(result)
        self.mock_session.delete.assert_called_once_with(mock_job)
        self.mock_session.commit.assert_called_once()

    # Chat session operations tests
    def test_create_chat_session(self):
        """Test creating a chat session."""
        # Create mock chat session
        mock_session = MagicMock(spec=ChatSession)
        mock_session.id = "test-session-id"
        mock_session.title = "Test Chat"
        
        # Set up the mock to intercept ChatSession instantiation
        with patch('app.providers.rel_db.ChatSession') as mock_chat_session_class:
            mock_chat_session_class.return_value = mock_session
            
            # Call the method to test
            result = self.test_db.create_chat_session(
                self.mock_session,
                user_id="test-user-id",
                session_s3_path="test/path/chat.json",
                title="Test Chat",
                summary="This is a test chat session."
            )
            
            # Verify results
            self.mock_session.add.assert_called_once()
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_session)
            # Verify ChatSession was instantiated with correct params
            mock_chat_session_class.assert_called_once_with(
                user_id="test-user-id",
                session_s3_path="test/path/chat.json",
                title="Test Chat",
                summary="This is a test chat session."
            )
        
    def test_get_chat_session(self):
        """Test getting a chat session by ID."""
        # Create a mock chat session
        mock_session = MagicMock(spec=ChatSession)
        mock_session.id = "test-session-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_session
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_chat_session(self.mock_session, "test-session-id")
        
        # Verify results
        self.assertEqual(result, mock_session)
        self.mock_session.query.assert_called_once_with(ChatSession)
        
    def test_get_user_chat_sessions(self):
        """Test getting all chat sessions for a user."""
        # Create mock sessions list
        mock_session1 = MagicMock(spec=ChatSession)
        mock_session1.id = "test-session-1"
        mock_session2 = MagicMock(spec=ChatSession)
        mock_session2.id = "test-session-2"
        mock_sessions = [mock_session1, mock_session2]
        
        # Set up the mock for session.query().filter().all()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_sessions
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.get_user_chat_sessions(self.mock_session, "test-user-id")
        
        # Verify results
        self.assertEqual(result, mock_sessions)
        self.mock_session.query.assert_called_once_with(ChatSession)
        
    def test_update_chat_session(self):
        """Test updating a chat session."""
        # Create a mock chat session
        mock_session = MagicMock(spec=ChatSession)
        mock_session.id = "test-session-id"
        mock_session.title = "Old Title"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_session
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test with new values
        result = self.test_db.update_chat_session(
            self.mock_session,
            "test-session-id",
            title="New Title",
            summary="Updated summary",
            processed_for_training=True
        )
        
        # Verify results
        self.assertEqual(result, mock_session)
        self.assertEqual(mock_session.title, "New Title")
        self.assertEqual(mock_session.summary, "Updated summary")
        self.assertEqual(mock_session.processed_for_training, True)
        self.mock_session.commit.assert_called_once()
        
    def test_delete_chat_session(self):
        """Test deleting a chat session."""
        # Create a mock chat session
        mock_session = MagicMock(spec=ChatSession)
        mock_session.id = "test-session-id"
        
        # Set up the mock for session.query().filter().first()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_session
        self.mock_session.query.return_value = mock_query
        
        # Call the method to test
        result = self.test_db.delete_chat_session(self.mock_session, "test-session-id")
        
        # Verify results
        self.assertTrue(result)
        self.mock_session.delete.assert_called_once_with(mock_session)
        self.mock_session.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main() 