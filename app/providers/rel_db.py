import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import uuid

from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime
from sqlalchemy import Integer, Boolean, Text, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    """User table model."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class Document(Base):
    """Document metadata table model."""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    filename = Column(Text, nullable=False)
    content_type = Column(Text, nullable=False)
    s3_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    processed = Column(Boolean, default=False)
    chunk_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="documents")

class TrainingJob(Base):
    """Training job metadata table model."""
    __tablename__ = "training_jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), nullable=False)  # 'queued', 'processing', 'completed', 'failed'
    attempt = Column(Integer, default=1)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    lora_path = Column(Text)
    error = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="training_jobs")

class ChatSession(Base):
    """Chat session metadata table model."""
    __tablename__ = "chat_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(Text)
    summary = Column(Text)
    session_s3_path = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    processed_for_training = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")

class RelationalDB:
    """
    A class for interacting with PostgreSQL relational database.
    This adapter provides methods to store and retrieve structured data.
    """
    
    def __init__(self, 
                 host: Optional[str] = None,
                 port: Optional[str] = None,
                 database: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 connection_string: Optional[str] = None):
        """
        Initialize the PostgreSQL database connection.
        
        Args:
            host: Database host (defaults to env var DB_HOST)
            port: Database port (defaults to env var DB_PORT)
            database: Database name (defaults to env var DB_NAME)
            user: Database user (defaults to env var DB_USER)
            password: Database password (defaults to env var DB_PASSWORD)
            connection_string: Full connection string if provided directly
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            host = host or os.environ.get('DB_HOST')
            port = port or os.environ.get('DB_PORT')
            database = database or os.environ.get('DB_NAME')
            user = user or os.environ.get('DB_USER')
            password = password or os.environ.get('DB_PASSWORD')
            
            if not all([host, port, database, user, password]):
                raise ValueError("Missing required PostgreSQL configuration")
            
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        # Initialize engine and session
        self.engine = create_engine(self.connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_db_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close_db_session(self, session: Session) -> None:
        """Close a database session."""
        session.close()
    
    # User methods
    def create_user(self, session: Session, user_id: Optional[str] = None) -> User:
        """
        Create a new user.
        
        Args:
            session: Database session
            user_id: Optional user ID (defaults to generated UUID)
            
        Returns:
            Created user object
        """
        try:
            user = User(id=user_id or str(uuid.uuid4()))
            session.add(user)
            session.flush()  # Flush to get errors before commit
            session.commit()
            session.refresh(user)
            return user
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    def get_user(self, session: Session, user_id: str) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            session: Database session
            user_id: User ID to retrieve
            
        Returns:
            User object if found, None otherwise
        """
        try:
            return session.query(User).filter(User.id == user_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user {user_id}: {e}")
            raise
    
    def get_or_create_user(self, session: Session, user_id: str) -> User:
        """
        Get an existing user by ID or create a new one if it doesn't exist.
        
        Args:
            session: Database session
            user_id: User ID to retrieve or create
            
        Returns:
            User object
        """
        try:
            # First try to get the user
            user = session.query(User).filter(User.id == user_id).first()
            
            # If the user doesn't exist, create a new one
            if not user:
                logger.info(f"User {user_id} not found, creating new user")
                user = User(id=user_id)
                session.add(user)
                session.flush()  # Flush to get errors before commit
                session.commit()
                session.refresh(user)
            else:
                logger.info(f"Found existing user with ID: {user_id}")
                
            return user
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error getting or creating user: {e}")
            raise
    
    # Document methods
    def create_document(self, 
                      session: Session, 
                      user_id: str, 
                      filename: str, 
                      content_type: str, 
                      s3_path: str) -> Document:
        """
        Create a new document record.
        
        Args:
            session: Database session
            user_id: User ID the document belongs to
            filename: Original filename
            content_type: MIME type of the document
            s3_path: Path to the document in Wasabi S3
            
        Returns:
            Created document object
        """
        try:
            document = Document(
                user_id=user_id,
                filename=filename,
                content_type=content_type,
                s3_path=s3_path
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating document for user {user_id}: {e}")
            raise
    
    def get_document(self, session: Session, document_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            session: Database session
            document_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            return session.query(Document).filter(Document.id == document_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise
    
    def get_user_documents(self, session: Session, user_id: str) -> List[Document]:
        """
        Get all documents for a user.
        
        Args:
            session: Database session
            user_id: User ID to get documents for
            
        Returns:
            List of document objects
        """
        try:
            return session.query(Document).filter(Document.user_id == user_id).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting documents for user {user_id}: {e}")
            raise
    
    def update_document_processed(self, 
                                session: Session, 
                                document_id: str, 
                                processed: bool, 
                                chunk_count: int) -> Optional[Document]:
        """
        Update document processing status.
        
        Args:
            session: Database session
            document_id: Document ID to update
            processed: Whether the document has been processed
            chunk_count: Number of chunks generated
            
        Returns:
            Updated document object if found, None otherwise
        """
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processed = processed
                document.chunk_count = chunk_count
                session.commit()
                session.refresh(document)
            return document
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating document {document_id}: {e}")
            raise
    
    def delete_document(self, session: Session, document_id: str) -> bool:
        """
        Delete a document record.
        
        Args:
            session: Database session
            document_id: Document ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                session.delete(document)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    # Training job methods
    def create_training_job(self, session: Session, user_id: str) -> TrainingJob:
        """
        Create a new training job.
        
        Args:
            session: Database session
            user_id: User ID the job belongs to
            
        Returns:
            Created training job object
        """
        try:
            job = TrainingJob(
                user_id=user_id,
                status="queued"
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating training job for user {user_id}: {e}")
            raise
    
    def get_training_job(self, session: Session, job_id: str) -> Optional[TrainingJob]:
        """
        Get a training job by ID.
        
        Args:
            session: Database session
            job_id: Job ID to retrieve
            
        Returns:
            Training job object if found, None otherwise
        """
        try:
            return session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting training job {job_id}: {e}")
            raise
    
    def get_user_training_jobs(self, session: Session, user_id: str) -> List[TrainingJob]:
        """
        Get all training jobs for a user.
        
        Args:
            session: Database session
            user_id: User ID to get jobs for
            
        Returns:
            List of training job objects
        """
        try:
            return session.query(TrainingJob).filter(TrainingJob.user_id == user_id).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting training jobs for user {user_id}: {e}")
            raise
    
    def update_training_job_status(self, 
                                 session: Session, 
                                 job_id: str, 
                                 status: str, 
                                 **kwargs) -> Optional[TrainingJob]:
        """
        Update a training job's status.
        
        Args:
            session: Database session
            job_id: Training job ID
            status: New status value ('queued', 'processing', 'completed', 'failed')
            **kwargs: Additional fields to update (lora_path, error, etc.)
            
        Returns:
            Updated training job object if found, None otherwise
        """
        try:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return None
                
            job.status = status
            
            # Update timestamps based on status
            if status == 'processing':
                job.started_at = datetime.now(timezone.utc)
            elif status in ['completed', 'failed']:
                job.completed_at = datetime.now(timezone.utc)
            
            # Update other fields from kwargs
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            session.commit()
            session.refresh(job)
            return job
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating training job {job_id}: {e}")
            raise
    
    def delete_training_job(self, session: Session, job_id: str) -> bool:
        """
        Delete a training job record.
        
        Args:
            session: Database session
            job_id: Job ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                session.delete(job)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting training job {job_id}: {e}")
            raise
    
    # Chat session methods
    def create_chat_session(self, 
                          session: Session, 
                          user_id: str, 
                          session_s3_path: str, 
                          title: Optional[str] = None, 
                          summary: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session record.
        
        Args:
            session: Database session
            user_id: User ID the chat belongs to
            session_s3_path: Path to the chat session files in Wasabi S3
            title: Optional chat title
            summary: Optional chat summary
            
        Returns:
            Created chat session object
        """
        try:
            chat_session = ChatSession(
                user_id=user_id,
                session_s3_path=session_s3_path,
                title=title,
                summary=summary
            )
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            return chat_session
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating chat session for user {user_id}: {e}")
            raise
    
    def get_chat_session(self, session: Session, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.
        
        Args:
            session: Database session
            session_id: Chat session ID to retrieve
            
        Returns:
            Chat session object if found, None otherwise
        """
        try:
            return session.query(ChatSession).filter(ChatSession.id == session_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting chat session {session_id}: {e}")
            raise
    
    def get_user_chat_sessions(self, session: Session, user_id: str) -> List[ChatSession]:
        """
        Get all chat sessions for a user.
        
        Args:
            session: Database session
            user_id: User ID to get chat sessions for
            
        Returns:
            List of chat session objects
        """
        try:
            return session.query(ChatSession).filter(ChatSession.user_id == user_id).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting chat sessions for user {user_id}: {e}")
            raise
    
    def update_chat_session(self, 
                          session: Session, 
                          session_id: str, 
                          **kwargs) -> Optional[ChatSession]:
        """
        Update chat session metadata.
        
        Args:
            session: Database session
            session_id: Chat session ID to update
            **kwargs: Fields to update (title, summary, processed_for_training)
            
        Returns:
            Updated chat session object if found, None otherwise
        """
        try:
            chat_session = session.query(ChatSession).filter(ChatSession.id == session_id).first()
            if chat_session:
                for key, value in kwargs.items():
                    if hasattr(chat_session, key):
                        setattr(chat_session, key, value)
                
                session.commit()
                session.refresh(chat_session)
            return chat_session
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating chat session {session_id}: {e}")
            raise
    
    def delete_chat_session(self, session: Session, session_id: str) -> bool:
        """
        Delete a chat session record.
        
        Args:
            session: Database session
            session_id: Chat session ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            chat_session = session.query(ChatSession).filter(ChatSession.id == session_id).first()
            if chat_session:
                session.delete(chat_session)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting chat session {session_id}: {e}")
            raise 