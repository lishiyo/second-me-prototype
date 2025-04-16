"""
SQL migrations for the L1 database schema.
This module contains the SQL statements needed to create the L1 tables in PostgreSQL.
"""

CREATE_TOPICS_TABLE = """
CREATE TABLE IF NOT EXISTS topics (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) REFERENCES users(id),
  name TEXT NOT NULL,
  summary TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  s3_path TEXT NOT NULL -- Path to detailed data in Wasabi
);
"""

CREATE_CLUSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS clusters (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) REFERENCES users(id),
  topic_id VARCHAR(36) REFERENCES topics(id),
  name TEXT,
  summary TEXT,
  document_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  s3_path TEXT NOT NULL -- Path to detailed data in Wasabi
);
"""

CREATE_CLUSTER_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS cluster_documents (
  cluster_id VARCHAR(36) REFERENCES clusters(id),
  document_id VARCHAR(36) REFERENCES documents(id),
  similarity_score FLOAT,
  PRIMARY KEY (cluster_id, document_id)
);
"""

CREATE_SHADES_TABLE = """
CREATE TABLE IF NOT EXISTS shades (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) REFERENCES users(id),
  name TEXT NOT NULL,
  summary TEXT,
  confidence FLOAT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  s3_path TEXT NOT NULL -- Path to detailed data in Wasabi
);
"""

# Migration to add new L1Shade fields
ALTER_SHADES_TABLE = """
ALTER TABLE shades
  ADD COLUMN IF NOT EXISTS aspect TEXT,
  ADD COLUMN IF NOT EXISTS icon TEXT,
  ADD COLUMN IF NOT EXISTS desc_second_view TEXT,
  ADD COLUMN IF NOT EXISTS desc_third_view TEXT,
  ADD COLUMN IF NOT EXISTS content_second_view TEXT,
  ADD COLUMN IF NOT EXISTS content_third_view TEXT;

-- Update existing records to set content_third_view from summary
UPDATE shades
  SET content_third_view = summary
  WHERE content_third_view IS NULL AND summary IS NOT NULL;
"""

CREATE_SHADE_CLUSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS shade_clusters (
  shade_id VARCHAR(36) REFERENCES shades(id),
  cluster_id VARCHAR(36) REFERENCES clusters(id),
  PRIMARY KEY (shade_id, cluster_id)
);
"""

CREATE_GLOBAL_BIOGRAPHIES_TABLE = """
CREATE TABLE IF NOT EXISTS global_biographies (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) REFERENCES users(id),
  content TEXT NOT NULL,
  content_third_view TEXT NOT NULL,
  summary TEXT NOT NULL,
  summary_third_view TEXT NOT NULL,
  confidence FLOAT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  version INTEGER NOT NULL
);
"""

CREATE_STATUS_BIOGRAPHIES_TABLE = """
CREATE TABLE IF NOT EXISTS status_biographies (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) REFERENCES users(id),
  content TEXT NOT NULL,
  content_third_view TEXT NOT NULL,
  summary TEXT NOT NULL,
  summary_third_view TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

CREATE_L1_VERSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS l1_versions (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) REFERENCES users(id),
  version INTEGER NOT NULL,
  status VARCHAR(20) NOT NULL, -- 'processing', 'completed', 'failed'
  started_at TIMESTAMP NOT NULL,
  completed_at TIMESTAMP,
  error TEXT,
  UNIQUE (user_id, version)
);
"""

# Add indexes for performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_topics_user_id ON topics(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_clusters_user_id ON clusters(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_clusters_topic_id ON clusters(topic_id);",
    "CREATE INDEX IF NOT EXISTS idx_cluster_documents_document_id ON cluster_documents(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_shades_user_id ON shades(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_global_biographies_user_id ON global_biographies(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_global_biographies_version ON global_biographies(version);",
    "CREATE INDEX IF NOT EXISTS idx_status_biographies_user_id ON status_biographies(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_l1_versions_user_id ON l1_versions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_l1_versions_user_version ON l1_versions(user_id, version);"
]

# Combine all migrations
ALL_MIGRATIONS = [
    CREATE_TOPICS_TABLE,
    CREATE_CLUSTERS_TABLE,
    CREATE_CLUSTER_DOCUMENTS_TABLE,
    CREATE_SHADES_TABLE,
    ALTER_SHADES_TABLE,
    CREATE_SHADE_CLUSTERS_TABLE,
    CREATE_GLOBAL_BIOGRAPHIES_TABLE,
    CREATE_STATUS_BIOGRAPHIES_TABLE,
    CREATE_L1_VERSIONS_TABLE
] + CREATE_INDEXES

def get_migration_sql():
    """Get all SQL statements for migration."""
    return ALL_MIGRATIONS 