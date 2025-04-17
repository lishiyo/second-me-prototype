from sqlalchemy import Column, String, DateTime, Integer, Float, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base

from datetime import datetime, timezone
import uuid
# Import Document and Base from rel_db to ensure we use the same metadata
from app.providers.rel_db import Document, Base

def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())

class L1Topic(Base):
    """SQLAlchemy model for topics table."""
    __tablename__ = "topics"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), 
                      onupdate=lambda: datetime.now(timezone.utc))
    s3_path = Column(Text, nullable=False)  # Path to detailed data in Wasabi
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=True)  # Link to L1Version
    
    # Relationships
    clusters = relationship("L1Cluster", back_populates="topic", cascade="all, delete-orphan")
    version_info = relationship("L1Version", back_populates="topics")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "s3_path": self.s3_path,
            "version": self.version
        }


class L1Cluster(Base):
    """SQLAlchemy model for clusters table."""
    __tablename__ = "clusters"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    topic_id = Column(String(36), ForeignKey("topics.id"), nullable=True)
    name = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    document_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), 
                      onupdate=lambda: datetime.now(timezone.utc))
    s3_path = Column(Text, nullable=False)  # Path to detailed data in Wasabi
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=True)  # Link to L1Version
    
    # Relationships
    topic = relationship("L1Topic", back_populates="clusters")
    cluster_documents = relationship("L1ClusterDocument", back_populates="cluster", cascade="all, delete-orphan")
    shade_clusters = relationship("L1ShadeCluster", back_populates="cluster", cascade="all, delete-orphan")
    version_info = relationship("L1Version", back_populates="clusters")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "topic_id": self.topic_id,
            "name": self.name,
            "summary": self.summary,
            "document_count": self.document_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "s3_path": self.s3_path,
            "version": self.version
        }


class L1ClusterDocument(Base):
    """SQLAlchemy model for cluster-document junction table."""
    __tablename__ = "cluster_documents"
    
    cluster_id = Column(String(36), ForeignKey("clusters.id"), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), primary_key=True)
    similarity_score = Column(Float, nullable=True)
    
    # Relationships
    cluster = relationship("L1Cluster", back_populates="cluster_documents")
    document = relationship(Document)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "document_id": self.document_id,
            "similarity_score": self.similarity_score
        }


class L1Shade(Base):
    """SQLAlchemy model for shades table."""
    __tablename__ = "shades"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), 
                      onupdate=lambda: datetime.now(timezone.utc))
    s3_path = Column(Text, nullable=False)  # Path to detailed data in Wasabi
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=True)  # Link to L1Version
    
    # New fields to match L1Shade model
    aspect = Column(Text, nullable=True)
    icon = Column(Text, nullable=True)
    desc_second_view = Column(Text, nullable=True)
    desc_third_view = Column(Text, nullable=True)
    content_second_view = Column(Text, nullable=True)
    content_third_view = Column(Text, nullable=True)
    
    # Relationships
    shade_clusters = relationship("L1ShadeCluster", back_populates="shade", cascade="all, delete-orphan")
    version_info = relationship("L1Version", back_populates="shades")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "summary": self.summary,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "s3_path": self.s3_path,
            "version": self.version,
            "aspect": self.aspect,
            "icon": self.icon,
            "desc_second_view": self.desc_second_view,
            "desc_third_view": self.desc_third_view,
            "content_second_view": self.content_second_view,
            "content_third_view": self.content_third_view
        }


class L1ShadeCluster(Base):
    """SQLAlchemy model for shade-cluster junction table."""
    __tablename__ = "shade_clusters"
    
    shade_id = Column(String(36), ForeignKey("shades.id"), primary_key=True)
    cluster_id = Column(String(36), ForeignKey("clusters.id"), primary_key=True)
    
    # Relationships
    shade = relationship("L1Shade", back_populates="shade_clusters")
    cluster = relationship("L1Cluster", back_populates="shade_clusters")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "shade_id": self.shade_id,
            "cluster_id": self.cluster_id
        }


class L1GlobalBiography(Base):
    """SQLAlchemy model for global biographies table."""
    __tablename__ = "global_biographies"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_third_view = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    summary_third_view = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), 
                      onupdate=lambda: datetime.now(timezone.utc))
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=False)
    
    # Add relationship to version_info
    version_info = relationship("L1Version", back_populates="global_biographies")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "content_third_view": self.content_third_view,
            "summary": self.summary,
            "summary_third_view": self.summary_third_view,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version
        }


class L1StatusBiography(Base):
    """SQLAlchemy model for status biographies table."""
    __tablename__ = "status_biographies"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_third_view = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    summary_third_view = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=True)
    
    # Add relationship to version_info
    version_info = relationship("L1Version", back_populates="status_biographies")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "content_third_view": self.content_third_view,
            "summary": self.summary,
            "summary_third_view": self.summary_third_view,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "version": self.version
        }


class L1Version(Base):
    """SQLAlchemy model for L1 processing versions."""
    __tablename__ = "l1_versions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    version = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)  # 'processing', 'completed', 'failed'
    started_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
    
    # Relationships to other L1 entities
    global_biographies = relationship("L1GlobalBiography", back_populates="version_info")
    status_biographies = relationship("L1StatusBiography", back_populates="version_info")
    clusters = relationship("L1Cluster", back_populates="version_info")
    shades = relationship("L1Shade", back_populates="version_info")
    topics = relationship("L1Topic", back_populates="version_info")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "version": self.version,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        } 