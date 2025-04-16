-- Migration script to add version columns to L1 tables
-- Date: 2025-04-20

-- Add version column to clusters table
ALTER TABLE clusters 
ADD version INTEGER;

-- Add foreign key constraint for clusters.version
ALTER TABLE clusters
ADD CONSTRAINT fk_clusters_version
FOREIGN KEY (user_id, version)
REFERENCES l1_versions(user_id, version);

-- Add version column to shades table
ALTER TABLE shades
ADD version INTEGER;

-- Add foreign key constraint for shades.version
ALTER TABLE shades
ADD CONSTRAINT fk_shades_version
FOREIGN KEY (user_id, version)
REFERENCES l1_versions(user_id, version);

-- Add version column to topics table
ALTER TABLE topics
ADD version INTEGER;

-- Add foreign key constraint for topics.version
ALTER TABLE topics
ADD CONSTRAINT fk_topics_version
FOREIGN KEY (user_id, version)
REFERENCES l1_versions(user_id, version);

-- Create index on clusters.version for performance
CREATE INDEX idx_clusters_version ON clusters(version);

-- Create index on shades.version for performance
CREATE INDEX idx_shades_version ON shades(version);

-- Create index on topics.version for performance
CREATE INDEX idx_topics_version ON topics(version);

-- Update any existing records to associate with latest version (optional)
-- The below is commented out as it requires determining the appropriate version logic
-- UPDATE clusters SET version = (SELECT MAX(version) FROM l1_versions WHERE user_id = clusters.user_id);
-- UPDATE shades SET version = (SELECT MAX(version) FROM l1_versions WHERE user_id = shades.user_id);
-- UPDATE topics SET version = (SELECT MAX(version) FROM l1_versions WHERE user_id = topics.user_id);

-- Record migration in migrations table (if you have one)
-- INSERT INTO migrations (name, applied_at) VALUES ('add_version_to_l1_tables', NOW()); 