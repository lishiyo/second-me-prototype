"""
PostgresAdapter for L1 layer.

This module provides an adapter for interacting with the PostgreSQL database 
for L1 data including topics, clusters, shades, and biographies.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from sqlalchemy import text, select, and_
from sqlalchemy.orm import Session
import json

from app.providers.rel_db import RelationalDB, Document
from app.models.l1.db_models import (
    L1Topic, L1Cluster, L1ClusterDocument, L1Shade, 
    L1ShadeCluster, L1GlobalBiography, L1StatusBiography,
    L1Version
)
from app.models.l1.topic import Topic, Cluster
from app.models.l1.shade import L1Shade as ShadeModel
from app.models.l1.bio import Bio

logger = logging.getLogger(__name__)

class PostgresAdapter:
    """
    Adapter for PostgreSQL database operations for L1 data.
    
    Provides methods for storing and retrieving L1 data including
    topics, clusters, shades, and biographies.
    """
    
    def __init__(self, rel_db: Optional[RelationalDB] = None):
        """
        Initialize the PostgresAdapter.
        
        Args:
            rel_db: RelationalDB instance. If None, a new instance is created.
        """
        self.rel_db = rel_db if rel_db else RelationalDB()
    
    def get_db_session(self) -> Session:
        """Get a database session."""
        return self.rel_db.get_db_session()
    
    def close_db_session(self, session: Session) -> None:
        """Close a database session."""
        self.rel_db.close_db_session(session)
    
    # Version tracking methods
    def create_version(self, user_id: str, version: int) -> L1Version:
        """
        Create a new L1 processing version record.
        
        Args:
            user_id: The user ID.
            version: The version number.
            
        Returns:
            The created L1Version object.
        """
        session = self.get_db_session()
        try:
            version_record = L1Version(
                user_id=user_id,
                version=version,
                status="processing",
                started_at=datetime.utcnow()
            )
            session.add(version_record)
            session.commit()
            return version_record
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating L1 version: {e}")
            raise
        finally:
            self.close_db_session(session)
    
    def update_version_status(self, user_id: str, version: int, status: str, 
                              error: Optional[str] = None) -> bool:
        """
        Update the status of an L1 processing version.
        
        Args:
            user_id: The user ID.
            version: The version number.
            status: The new status ("processing", "completed", "failed").
            error: Optional error message if status is "failed".
            
        Returns:
            True if update was successful, False otherwise.
        """
        session = self.get_db_session()
        try:
            version_record = session.query(L1Version).filter(
                and_(
                    L1Version.user_id == user_id,
                    L1Version.version == version
                )
            ).first()
            
            if not version_record:
                logger.error(f"Version {version} not found for user {user_id}")
                return False
            
            version_record.status = status
            if status == "completed":
                version_record.completed_at = datetime.utcnow()
            elif status == "failed":
                version_record.error = error
                version_record.completed_at = datetime.utcnow()
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating L1 version status: {e}")
            return False
        finally:
            self.close_db_session(session)
    
    def get_latest_version(self, user_id: str) -> Optional[int]:
        """
        Get the latest L1 processing version for a user.
        
        Args:
            user_id: The user ID.
            
        Returns:
            The latest version number or None if no versions exist.
        """
        session = self.get_db_session()
        try:
            version_record = session.query(L1Version).filter(
                L1Version.user_id == user_id
            ).order_by(L1Version.version.desc()).first()
            
            return version_record.version if version_record else None
        except Exception as e:
            logger.error(f"Error getting latest L1 version: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    # Topic methods
    def create_topic(self, user_id: str, name: str, summary: str, s3_path: str) -> Optional[L1Topic]:
        """
        Create a new topic.
        
        Args:
            user_id: The user ID.
            name: Topic name.
            summary: Topic summary.
            s3_path: S3 path to the topic data.
            
        Returns:
            The created L1Topic object or None if creation failed.
        """
        session = self.get_db_session()
        try:
            topic = L1Topic(
                user_id=user_id,
                name=name,
                summary=summary,
                s3_path=s3_path,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(topic)
            session.commit()
            return topic
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating topic: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    def get_topics(self, user_id: str) -> List[L1Topic]:
        """
        Get all topics for a user.
        
        Args:
            user_id: The user ID.
            
        Returns:
            List of L1Topic objects.
        """
        session = self.get_db_session()
        try:
            topics = session.query(L1Topic).filter(L1Topic.user_id == user_id).all()
            return topics
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return []
        finally:
            self.close_db_session(session)
    
    def get_topic(self, topic_id: str) -> Optional[L1Topic]:
        """
        Get a topic by ID.
        
        Args:
            topic_id: The topic ID.
            
        Returns:
            L1Topic object or None if not found.
        """
        session = self.get_db_session()
        try:
            topic = session.query(L1Topic).filter(L1Topic.id == topic_id).first()
            return topic
        except Exception as e:
            logger.error(f"Error getting topic: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    # Cluster methods
    # def create_cluster(self, user_id: str, topic_id: str, name: str, 
    #                    summary: str, document_count: int, s3_path: str) -> Optional[L1Cluster]:
    #     """
    #     Create a new cluster.
        
    #     Args:
    #         user_id: The user ID.
    #         topic_id: The parent topic ID.
    #         name: Cluster name.
    #         summary: Cluster summary.
    #         document_count: Number of documents in the cluster.
    #         s3_path: S3 path to the cluster data.
            
    #     Returns:
    #         The created L1Cluster object or None if creation failed.
    #     """
    #     session = self.get_db_session()
    #     try:
    #         cluster = L1Cluster(
    #             user_id=user_id,
    #             topic_id=topic_id,
    #             name=name,
    #             summary=summary,
    #             document_count=document_count,
    #             s3_path=s3_path,
    #             created_at=datetime.utcnow(),
    #             updated_at=datetime.utcnow()
    #         )
    #         session.add(cluster)
    #         session.commit()
    #         return cluster
    #     except Exception as e:
    #         session.rollback()
    #         logger.error(f"Error creating cluster: {e}")
    #         return None
    #     finally:
    #         self.close_db_session(session)
    
    def add_document_to_cluster(self, cluster_id: str, document_id: str, 
                               similarity_score: float) -> bool:
        """
        Add a document to a cluster.
        
        Args:
            cluster_id: The cluster ID.
            document_id: The document ID.
            similarity_score: The similarity score between document and cluster.
            
        Returns:
            True if successful, False otherwise.
        """
        session = self.get_db_session()
        try:
            cluster_doc = L1ClusterDocument(
                cluster_id=cluster_id,
                document_id=document_id,
                similarity_score=similarity_score
            )
            session.add(cluster_doc)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding document to cluster: {e}")
            return False
        finally:
            self.close_db_session(session)
    
    def get_clusters(self, user_id: str, topic_id: Optional[str] = None) -> List[L1Cluster]:
        """
        Get clusters for a user, optionally filtered by topic.
        
        Args:
            user_id: The user ID.
            topic_id: Optional topic ID to filter clusters.
            
        Returns:
            List of L1Cluster objects.
        """
        session = self.get_db_session()
        try:
            query = session.query(L1Cluster).filter(L1Cluster.user_id == user_id)
            if topic_id:
                query = query.filter(L1Cluster.topic_id == topic_id)
            
            clusters = query.all()
            return clusters
        except Exception as e:
            logger.error(f"Error getting clusters: {e}")
            return []
        finally:
            self.close_db_session(session)
    
    def get_cluster(self, cluster_id: str) -> Optional[L1Cluster]:
        """
        Get a cluster by ID.
        
        Args:
            cluster_id: The cluster ID.
            
        Returns:
            L1Cluster object or None if not found.
        """
        session = self.get_db_session()
        try:
            cluster = session.query(L1Cluster).filter(L1Cluster.id == cluster_id).first()
            return cluster
        except Exception as e:
            logger.error(f"Error getting cluster: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    def get_cluster_documents(self, cluster_id: str) -> List[Tuple[str, float]]:
        """
        Get documents associated with a cluster.
        
        Args:
            cluster_id: The cluster ID.
            
        Returns:
            List of tuples (document_id, similarity_score).
        """
        session = self.get_db_session()
        try:
            docs = session.query(L1ClusterDocument).filter(
                L1ClusterDocument.cluster_id == cluster_id
            ).all()
            
            return [(doc.document_id, doc.similarity_score) for doc in docs]
        except Exception as e:
            logger.error(f"Error getting cluster documents: {e}")
            return []
        finally:
            self.close_db_session(session)
    
    # Shade methods
    # def create_shade(self, user_id: str, name: str, summary: str, 
    #                 confidence: float, s3_path: str) -> Optional[L1Shade]:
    #     """
    #     Create a new shade.
        
    #     Args:
    #         user_id: The user ID.
    #         name: Shade name.
    #         summary: Shade summary.
    #         confidence: Confidence score for the shade.
    #         s3_path: S3 path to the shade data.
            
    #     Returns:
    #         The created L1Shade object or None if creation failed.
    #     """
    #     session = self.get_db_session()
    #     try:
    #         shade = L1Shade(
    #             user_id=user_id,
    #             name=name,
    #             summary=summary,
    #             confidence=confidence,
    #             s3_path=s3_path,
    #             created_at=datetime.utcnow(),
    #             updated_at=datetime.utcnow()
    #         )
    #         session.add(shade)
    #         session.commit()
    #         return shade
    #     except Exception as e:
    #         session.rollback()
    #         logger.error(f"Error creating shade: {e}")
    #         return None
    #     finally:
    #         self.close_db_session(session)
    
    def add_cluster_to_shade(self, shade_id: str, cluster_id: str) -> bool:
        """
        Associate a cluster with a shade.
        
        Args:
            shade_id: The shade ID.
            cluster_id: The cluster ID.
            
        Returns:
            True if successful, False otherwise.
        """
        session = self.get_db_session()
        try:
            shade_cluster = L1ShadeCluster(
                shade_id=shade_id,
                cluster_id=cluster_id
            )
            session.add(shade_cluster)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding cluster to shade: {e}")
            return False
        finally:
            self.close_db_session(session)
    
    def get_shades(self, user_id: str) -> List[L1Shade]:
        """
        Get all shades for a user.
        
        Args:
            user_id: The user ID.
            
        Returns:
            List of L1Shade objects.
        """
        session = self.get_db_session()
        try:
            shades = session.query(L1Shade).filter(L1Shade.user_id == user_id).all()
            return shades
        except Exception as e:
            logger.error(f"Error getting shades: {e}")
            return []
        finally:
            self.close_db_session(session)
    
    def get_shade(self, shade_id: str) -> Optional[L1Shade]:
        """
        Get a shade by ID.
        
        Args:
            shade_id: The shade ID.
            
        Returns:
            L1Shade object or None if not found.
        """
        session = self.get_db_session()
        try:
            shade = session.query(L1Shade).filter(L1Shade.id == shade_id).first()
            return shade
        except Exception as e:
            logger.error(f"Error getting shade: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    def get_shade_clusters(self, shade_id: str) -> List[str]:
        """
        Get clusters associated with a shade.
        
        Args:
            shade_id: The shade ID.
            
        Returns:
            List of cluster IDs.
        """
        session = self.get_db_session()
        try:
            shade_clusters = session.query(L1ShadeCluster).filter(
                L1ShadeCluster.shade_id == shade_id
            ).all()
            
            return [sc.cluster_id for sc in shade_clusters]
        except Exception as e:
            logger.error(f"Error getting shade clusters: {e}")
            return []
        finally:
            self.close_db_session(session)
    
    # Biography methods
    # def create_global_biography(self, user_id: str, content: str, content_third_view: str,
    #                            summary: str, summary_third_view: str, confidence: float,
    #                            version: int) -> Optional[L1GlobalBiography]:
    #     """
    #     Create a new global biography.
        
    #     Args:
    #         user_id: The user ID.
    #         content: Biography content in first person.
    #         content_third_view: Biography content in third person.
    #         summary: Short summary in first person.
    #         summary_third_view: Short summary in third person.
    #         confidence: Confidence score for the biography.
    #         version: The version number.
            
    #     Returns:
    #         The created L1GlobalBiography object or None if creation failed.
    #     """
    #     session = self.get_db_session()
    #     try:
    #         bio = L1GlobalBiography(
    #             user_id=user_id,
    #             content=content,
    #             content_third_view=content_third_view,
    #             summary=summary,
    #             summary_third_view=summary_third_view,
    #             confidence=confidence,
    #             version=version,
    #             created_at=datetime.utcnow(),
    #             updated_at=datetime.utcnow()
    #         )
    #         session.add(bio)
    #         session.commit()
    #         return bio
    #     except Exception as e:
    #         session.rollback()
    #         logger.error(f"Error creating global biography: {e}")
    #         return None
    #     finally:
    #         self.close_db_session(session)
    
    def create_status_biography(self, user_id: str, content: str, content_third_view: str,
                               summary: str, summary_third_view: str) -> Optional[L1StatusBiography]:
        """
        Create a new status biography.
        
        Args:
            user_id: The user ID.
            content: Biography content in first person.
            content_third_view: Biography content in third person.
            summary: Short summary in first person.
            summary_third_view: Short summary in third person.
            
        Returns:
            The created L1StatusBiography object or None if creation failed.
        """
        session = self.get_db_session()
        try:
            bio = L1StatusBiography(
                user_id=user_id,
                content=content,
                content_third_view=content_third_view,
                summary=summary,
                summary_third_view=summary_third_view,
                created_at=datetime.utcnow()
            )
            session.add(bio)
            session.commit()
            return bio
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating status biography: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    def get_latest_global_biography(self, user_id: str) -> Optional[L1GlobalBiography]:
        """
        Get the latest global biography for a user.
        
        Args:
            user_id: The user ID.
            
        Returns:
            L1GlobalBiography object or None if not found.
        """
        session = self.get_db_session()
        try:
            bio = session.query(L1GlobalBiography).filter(
                L1GlobalBiography.user_id == user_id
            ).order_by(L1GlobalBiography.created_at.desc()).first()
            
            return bio
        except Exception as e:
            logger.error(f"Error getting latest global biography: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    def get_latest_status_biography(self, user_id: str) -> Optional[L1StatusBiography]:
        """
        Get the latest status biography for a user.
        
        Args:
            user_id: The user ID.
            
        Returns:
            L1StatusBiography object or None if not found.
        """
        session = self.get_db_session()
        try:
            bio = session.query(L1StatusBiography).filter(
                L1StatusBiography.user_id == user_id
            ).order_by(L1StatusBiography.created_at.desc()).first()
            
            return bio
        except Exception as e:
            logger.error(f"Error getting latest status biography: {e}")
            return None
        finally:
            self.close_db_session(session)
    
    # Domain model methods
    
    def create_topic_from_model(self, user_id: str, topic: Topic) -> Optional[L1Topic]:
        """
        Create a new topic from a Topic domain model.
        
        Args:
            user_id: The user ID.
            topic: Topic domain model.
            
        Returns:
            The created L1Topic object or None if creation failed.
        """
        s3_path = f"l1/topics/{user_id}/{topic.id}.json"
        return self.create_topic(
            user_id=user_id,
            name=topic.name,
            summary=topic.summary or "",
            s3_path=s3_path
        )
    
    # def create_cluster_from_model(self, user_id: str, cluster: Cluster) -> Optional[L1Cluster]:
    #     """
    #     Create a new cluster from a Cluster domain model.
        
    #     Args:
    #         user_id: The user ID.
    #         cluster: Cluster domain model.
            
    #     Returns:
    #         The created L1Cluster object or None if creation failed.
    #     """
    #     s3_path = f"l1/clusters/{user_id}/{cluster.id}.json"
    #     db_cluster = self.create_cluster(
    #         user_id=user_id,
    #         topic_id=cluster.topic_id or "",
    #         name=cluster.name or "",
    #         summary=cluster.summary or "",
    #         document_count=cluster.document_count,
    #         s3_path=s3_path
    #     )
        
    #     # Add documents to cluster
    #     if db_cluster:
    #         for document_id in cluster.document_ids:
    #             # Use default similarity score of 1.0 if not available
    #             similarity_score = 1.0
    #             self.add_document_to_cluster(db_cluster.id, document_id, similarity_score)
        
    #     return db_cluster
    
    # def create_shade_from_model(self, user_id: str, shade: ShadeModel) -> Optional[L1Shade]:
    #     """
    #     Create a L1Shade database record from a domain model.
        
    #     Args:
    #         user_id: The user ID.
    #         shade: ShadeModel domain model.
            
    #     Returns:
    #         The created L1Shade database object or None if creation failed.
    #     """
    #     session = self.get_db_session()
    #     try:
    #         db_shade = L1Shade(
    #             id=shade.id,
    #             user_id=user_id,
    #             name=shade.name,
    #             summary=shade.summary,
    #             confidence=shade.confidence,
    #             s3_path=shade.s3_path or "",
    #             created_at=datetime.fromisoformat(shade.created_at) if isinstance(shade.created_at, str) else shade.created_at,
    #             updated_at=datetime.fromisoformat(shade.updated_at) if isinstance(shade.updated_at, str) else shade.updated_at,
    #             aspect=shade.aspect,
    #             icon=shade.icon,
    #             desc_second_view=shade.desc_second_view,
    #             desc_third_view=shade.desc_third_view,
    #             content_second_view=shade.content_second_view,
    #             content_third_view=shade.content_third_view
    #         )
    #         session.add(db_shade)
    #         session.commit()
    #         return db_shade
    #     except Exception as e:
    #         session.rollback()
    #         logger.error(f"Error creating shade from model: {e}")
    #         return None
    #     finally:
    #         self.close_db_session(session)
    
    # def create_global_biography_from_model(self, user_id: str, bio: Bio, version: int) -> Optional[L1GlobalBiography]:
    #     """
    #     Create a new global biography from a Bio domain model.
        
    #     Args:
    #         user_id: The user ID.
    #         bio: Bio domain model.
    #         version: Version number.
            
    #     Returns:
    #         The created L1GlobalBiography object or None if creation failed.
    #     """
    #     return self.create_global_biography(
    #         user_id=user_id,
    #         content=bio.content_first_view,
    #         content_third_view=bio.content_third_view,
    #         summary=bio.summary_first_view,
    #         summary_third_view=bio.summary_third_view,
    #         confidence=bio.confidence,
    #         version=version
    #     )
    
    def create_status_biography_from_model(self, user_id: str, bio: Bio) -> Optional[L1StatusBiography]:
        """
        Create a new status biography from a Bio domain model.
        
        Args:
            user_id: The user ID.
            bio: Bio domain model.
            
        Returns:
            The created L1StatusBiography object or None if creation failed.
        """
        return self.create_status_biography(
            user_id=user_id,
            content=bio.content_first_view,
            content_third_view=bio.content_third_view,
            summary=bio.summary_first_view,
            summary_third_view=bio.summary_third_view
        )
    
    def convert_to_topic_model(self, db_topic: L1Topic) -> Topic:
        """
        Convert an L1Topic database model to a Topic domain model.
        
        Args:
            db_topic: L1Topic database model.
            
        Returns:
            Topic domain model.
        """
        return Topic(
            id=db_topic.id,
            name=db_topic.name,
            summary=db_topic.summary,
            document_ids=[],  # Would need to query for this
            created_at=db_topic.created_at,
            updated_at=db_topic.updated_at,
            metadata={},
            s3_path=db_topic.s3_path
        )
    
    def convert_to_shade_model(self, db_shade: L1Shade) -> ShadeModel:
        """
        Convert a database shade to a domain model.
        
        Args:
            db_shade: L1Shade database object.
            
        Returns:
            ShadeModel domain model.
        """
        return ShadeModel(
            id=db_shade.id,
            user_id=db_shade.user_id,
            name=db_shade.name,
            summary=db_shade.summary,
            confidence=db_shade.confidence,
            created_at=db_shade.created_at.isoformat() if db_shade.created_at else None,
            updated_at=db_shade.updated_at.isoformat() if db_shade.updated_at else None,
            s3_path=db_shade.s3_path,
            aspect=db_shade.aspect,
            icon=db_shade.icon,
            desc_second_view=db_shade.desc_second_view,
            desc_third_view=db_shade.desc_third_view,
            content_second_view=db_shade.content_second_view,
            content_third_view=db_shade.content_third_view
        )
    
    def convert_to_cluster_model(self, db_cluster: L1Cluster) -> Cluster:
        """
        Convert an L1Cluster database model to a Cluster domain model.
        
        Args:
            db_cluster: L1Cluster database model.
            
        Returns:
            Cluster domain model.
        """
        # Get documents in this cluster
        documents = self.get_cluster_documents(db_cluster.id)
        document_ids = [doc_id for doc_id, _ in documents]
        
        return Cluster(
            id=db_cluster.id,
            topic_id=db_cluster.topic_id,
            name=db_cluster.name,
            summary=db_cluster.summary,
            memory_list=[],  # Would need actual Memory objects
            created_at=db_cluster.created_at,
            updated_at=db_cluster.updated_at,
            metadata={},
            s3_path=db_cluster.s3_path
        )
    
    def convert_to_bio_model(self, db_bio: Union[L1GlobalBiography, L1StatusBiography]) -> Bio:
        """
        Convert a biography database model to a Bio domain model.
        
        Args:
            db_bio: L1GlobalBiography or L1StatusBiography database model.
            
        Returns:
            Bio domain model.
        """
        return Bio(
            content_first_view=db_bio.content,
            content_third_view=db_bio.content_third_view,
            summary_first_view=db_bio.summary,
            summary_third_view=db_bio.summary_third_view,
            confidence=getattr(db_bio, 'confidence', 0.0),
            shades_list=[]  # Would need to query for this
        )
    
    def get_topic_model(self, topic_id: str) -> Optional[Topic]:
        """
        Get a Topic domain model by ID.
        
        Args:
            topic_id: The topic ID.
            
        Returns:
            Topic domain model or None if not found.
        """
        db_topic = self.get_topic(topic_id)
        if db_topic:
            return self.convert_to_topic_model(db_topic)
        return None
    
    def get_shade_model(self, shade_id: str) -> Optional[ShadeModel]:
        """
        Get a shade domain model by ID.
        
        Args:
            shade_id: The shade ID.
            
        Returns:
            ShadeModel domain model or None if not found.
        """
        db_shade = self.get_shade(shade_id)
        if db_shade:
            return self.convert_to_shade_model(db_shade)
        return None
    
    def get_cluster_model(self, cluster_id: str) -> Optional[Cluster]:
        """
        Get a Cluster domain model by ID.
        
        Args:
            cluster_id: The cluster ID.
            
        Returns:
            Cluster domain model or None if not found.
        """
        db_cluster = self.get_cluster(cluster_id)
        if db_cluster:
            return self.convert_to_cluster_model(db_cluster)
        return None
    
    def get_latest_global_bio_model(self, user_id: str) -> Optional[Bio]:
        """
        Get the latest global biography as a Bio domain model.
        
        Args:
            user_id: The user ID.
            
        Returns:
            Bio domain model or None if not found.
        """
        db_bio = self.get_latest_global_biography(user_id)
        if db_bio:
            return self.convert_to_bio_model(db_bio)
        return None
    
    def get_latest_status_bio_model(self, user_id: str) -> Optional[Bio]:
        """
        Get the latest status biography as a Bio domain model.
        
        Args:
            user_id: The user ID.
            
        Returns:
            Bio domain model or None if not found.
        """
        db_bio = self.get_latest_status_biography(user_id)
        if db_bio:
            return self.convert_to_bio_model(db_bio)
        return None
    
    def get_documents_with_l0(self, user_id: str) -> List[Any]:
        """
        Get all documents with L0 data for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            List of document objects with L0 data
        """
        session = self.get_db_session()
        try:
            # Query documents that have been processed and have insight/summary data
            documents = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.processed == True,
                    Document.chunk_count > 0
                )
            ).all()
            return documents
        except Exception as e:
            logger.error(f"Error getting documents with L0: {e}")
            return []
        finally:
            self.close_db_session(session)
    
    # New storage methods with version support
    
    def store_cluster(self, user_id: str, cluster_id: str, name: str, 
                     document_ids: List[str], version: int) -> bool:
        """
        Store cluster metadata with version information.
        
        Args:
            user_id: The user ID
            cluster_id: The cluster ID
            name: The cluster name
            document_ids: List of document IDs associated with this cluster
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_db_session()
        try:
            # First check if cluster already exists
            existing_cluster = session.query(L1Cluster).filter(
                L1Cluster.id == cluster_id
            ).first()
            
            if existing_cluster:
                # Update existing cluster
                existing_cluster.user_id = user_id
                existing_cluster.name = name
                existing_cluster.document_count = len(document_ids)
                existing_cluster.updated_at = datetime.utcnow()
                existing_cluster.version = version  # Update version
                
                # Delete existing document associations to recreate them
                session.query(L1ClusterDocument).filter(
                    L1ClusterDocument.cluster_id == cluster_id
                ).delete()
            else:
                # Create a new cluster record
                s3_path = f"l1/clusters/{user_id}/{cluster_id}.json"
                cluster = L1Cluster(
                    id=cluster_id,
                    user_id=user_id,
                    name=name,
                    document_count=len(document_ids),
                    s3_path=s3_path,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    version=version  # Set version
                )
                session.add(cluster)
            
            # Add document associations with default similarity score
            for doc_id in document_ids:
                cluster_doc = L1ClusterDocument(
                    cluster_id=cluster_id,
                    document_id=doc_id,
                    similarity_score=1.0  # Default similarity score
                )
                session.add(cluster_doc)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing cluster: {e}")
            return False
        finally:
            self.close_db_session(session)
    
    def store_chunk_topics(self, user_id: str, chunk_topics: Dict[str, Dict], version: int) -> bool:
        """
        Store chunk topics with version information.
        
        Args:
            user_id: The user ID
            chunk_topics: Dictionary mapping chunk IDs to topic data
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_db_session()
        try:
            # Store in a custom JSON table or another appropriate storage structure
            # For now, we'll use the L1Topic table with a special prefix to differentiate
            for chunk_id, topic_data in chunk_topics.items():
                topic_id = f"chunk_{chunk_id}"
                topic_name = topic_data.get("topic", "Unknown Topic")
                summary = json.dumps(topic_data)  # Store the full topic data as JSON string
                
                # Check if topic already exists
                existing_topic = session.query(L1Topic).filter(
                    L1Topic.id == topic_id
                ).first()
                
                if existing_topic:
                    # Update existing topic
                    existing_topic.name = topic_name
                    existing_topic.summary = summary
                    existing_topic.updated_at = datetime.utcnow()
                    existing_topic.version = version  # Update version
                else:
                    # Create a new topic record
                    s3_path = f"l1/chunk_topics/{user_id}/{topic_id}.json"
                    topic = L1Topic(
                        id=topic_id,
                        user_id=user_id,
                        name=topic_name,
                        summary=summary,
                        s3_path=s3_path,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        version=version  # Set version
                    )
                    session.add(topic)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing chunk topics: {e}")
            return False
        finally:
            self.close_db_session(session)
    
    def store_shade(self, user_id: str, shade_id: str, name: str, summary: str,
                   confidence: float, source_clusters: List[str], version: int) -> bool:
        """
        Store shade metadata with version information.
        
        Args:
            user_id: The user ID
            shade_id: The shade ID
            name: The shade name
            summary: The shade summary
            confidence: The confidence score
            source_clusters: List of source cluster IDs
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_db_session()
        try:
            # Check if shade already exists
            existing_shade = session.query(L1Shade).filter(
                L1Shade.id == shade_id
            ).first()
            
            if existing_shade:
                # Update existing shade
                existing_shade.name = name
                existing_shade.summary = summary
                existing_shade.confidence = confidence
                existing_shade.updated_at = datetime.utcnow()
                existing_shade.version = version  # Update version
                
                # Delete existing cluster associations to recreate them
                session.query(L1ShadeCluster).filter(
                    L1ShadeCluster.shade_id == shade_id
                ).delete()
            else:
                # Create a new shade record
                s3_path = f"l1/shades/{user_id}/{shade_id}.json"
                shade = L1Shade(
                    id=shade_id,
                    user_id=user_id,
                    name=name,
                    summary=summary,
                    confidence=confidence,
                    s3_path=s3_path,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    version=version  # Set version
                )
                session.add(shade)
            
            # Add cluster associations
            for cluster_id in source_clusters:
                shade_cluster = L1ShadeCluster(
                    shade_id=shade_id,
                    cluster_id=cluster_id
                )
                session.add(shade_cluster)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing shade: {e}")
            return False
        finally:
            self.close_db_session(session)
    
    def store_global_biography(self, user_id: str, content: str, content_third_view: str,
                             summary: str, summary_third_view: str, version: int) -> bool:
        """
        Store global biography with version information.
        
        Args:
            user_id: The user ID
            content: Biography content in second person (for compatibility)
            content_third_view: Biography content in third person
            summary: Biography summary in second person (for compatibility)
            summary_third_view: Biography summary in third person
            version: The L1 version number
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_db_session()
        try:
            # Check if a biography for this version already exists
            existing_bio = session.query(L1GlobalBiography).filter(
                and_(
                    L1GlobalBiography.user_id == user_id,
                    L1GlobalBiography.version == version
                )
            ).first()
            
            if existing_bio:
                # Update existing biography
                existing_bio.content = content
                existing_bio.content_third_view = content_third_view
                existing_bio.summary = summary
                existing_bio.summary_third_view = summary_third_view
                existing_bio.updated_at = datetime.utcnow()
            else:
                # Create a new biography record
                bio = L1GlobalBiography(
                    user_id=user_id,
                    content=content,
                    content_third_view=content_third_view,
                    summary=summary,
                    summary_third_view=summary_third_view,
                    confidence=0.8,  # Default confidence
                    version=version,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(bio)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing global biography: {e}")
            return False
        finally:
            self.close_db_session(session) 