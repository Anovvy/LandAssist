-- Migration 002: LandAssist Prediction Lifecycle State Tracking
-- Supports Add Feature -> Undo (Ctrl+Z) -> Commit Changeset workflow

ALTER TABLE landuse_predictions 
ADD COLUMN IF NOT EXISTS status VARCHAR(16) NOT NULL DEFAULT 'unreviewed' 
    CHECK (status IN ('unreviewed', 'staged', 'accepted', 'rejected')),
ADD COLUMN IF NOT EXISTS reviewed_by VARCHAR(64) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ DEFAULT NULL,
ADD COLUMN IF NOT EXISTS osm_id BIGINT DEFAULT NULL;

-- Fast index for active prediction queries
CREATE INDEX IF NOT EXISTS idx_landuse_unreviewed 
    ON landuse_predictions (model_version, status);
