CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id UUID NOT NULL,                    -- уникальный id всего документа
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(2048) NOT NULL,              -- под размер твоей модели эмбеддингов
    metadata JSONB DEFAULT '{}'::jsonb,           -- сюда всё остальное
    page_number INTEGER,
    section TEXT,
    document_type TEXT,                           -- pdf, contract, manual и т.д.
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);

-- Индекс для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_embedding 
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_metadata 
    ON document_chunks USING gin (metadata);