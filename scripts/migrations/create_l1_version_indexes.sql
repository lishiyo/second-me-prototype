-- Migration script to create indexes for L1 version queries
-- Date: 2025-04-20

-- Create composite index on clusters (user_id, version) for fast filtering
CREATE INDEX idx_clusters_user_version ON clusters(user_id, version);

-- Create composite index on shades (user_id, version) for fast filtering
CREATE INDEX idx_shades_user_version ON shades(user_id, version);

-- Create composite index on topics (user_id, version) for fast filtering
CREATE INDEX idx_topics_user_version ON topics(user_id, version);

-- Create composite index on global_biographies (user_id, version) for fast filtering
CREATE INDEX idx_global_biographies_user_version ON global_biographies(user_id, version);

-- Create index on l1_versions (user_id, version) for fast version lookup
CREATE INDEX idx_l1_versions_user_version ON l1_versions(user_id, version);

-- Create index on l1_versions (user_id, status) for filtering by status
CREATE INDEX idx_l1_versions_user_status ON l1_versions(user_id, status);

-- Record migration in migrations table (if you have one)
-- INSERT INTO migrations (name, applied_at) VALUES ('create_l1_version_indexes', NOW()); 