-- Reference schema for jobs table (SQLAlchemy create_all is primary for MVP).

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ NULL,
    finished_at TIMESTAMPTZ NULL,
    input_video_path TEXT NOT NULL,
    params JSONB NOT NULL DEFAULT '{}',
    progress JSONB NOT NULL DEFAULT '{}',
    result_path TEXT NULL,
    error_type VARCHAR(64) NULL,
    error_message TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at);
