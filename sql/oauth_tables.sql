-- Run once in Supabase SQL Editor (or any Postgres) if tables are missing.
-- OAuth PKCE (survives serverless cold starts / multi-instance)
CREATE TABLE IF NOT EXISTS oauth_pkce (
    state TEXT PRIMARY KEY,
    verifier TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Persisted X tokens (access + refresh)
CREATE TABLE IF NOT EXISTS oauth_tokens (
    provider TEXT PRIMARY KEY,
    access_token TEXT,
    refresh_token TEXT,
    updated_at TEXT
);
