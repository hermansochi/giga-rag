"""
src/database.py

Модуль работы с базой данных (PostgreSQL + pgvector).
Исправлена функция log_token_usage — теперь баланс сохраняется корректно.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import hashlib

from .config import settings


def get_db_connection():
    """
    Возвращает соединение к БД для текущей сессии Streamlit.
    Соединение создаётся один раз и переиспользуется.
    """
    if "db_connection" not in st.session_state:
        try:
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                cursor_factory=RealDictCursor,
            )
            st.session_state.db_connection = conn
        except Exception as e:
            st.error(f"❌ Не удалось подключиться к PostgreSQL: {e}")
            st.stop()

    return st.session_state.db_connection


def init_vector_db():
    """
    Инициализация базы данных.
    """
    if st.session_state.get("db_fully_initialized", False):
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Расширение vector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Таблица чанков
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id UUID NOT NULL,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    embedding VECTOR({settings.EMBEDDING_DIM}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    page_number INTEGER,
                    section TEXT,
                    document_type TEXT DEFAULT 'pdf',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(document_id, chunk_index)
                );
            """)

            # Таблица логов токенов
            cur.execute("""
                CREATE TABLE IF NOT EXISTS token_usage_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    total_tokens INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    precached_tokens INTEGER DEFAULT 0,
                    balance_entries JSONB
                );
            """)

            # Индексы
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_hash ON document_chunks (chunk_hash);
                CREATE INDEX IF NOT EXISTS idx_document_id ON document_chunks (document_id);
                CREATE INDEX IF NOT EXISTS idx_metadata_gin ON document_chunks USING gin (metadata);
            """)

            conn.commit()

        st.session_state.db_fully_initialized = True

    except Exception as e:
        st.error(f"❌ Ошибка инициализации базы данных: {e}")


def save_chunks(
    filename: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    document_type: str = "pdf"
) -> str:
    """
    Сохраняет только новые чанки. Не сохраняет дубликаты.
    """
    if metadata_list is None:
        metadata_list = [{} for _ in chunks]

    document_id = str(uuid.uuid4())
    embedding_model = settings.EMBEDDING_MODEL
    new_chunks_count = 0

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for i, (text, embedding, meta) in enumerate(zip(chunks, embeddings, metadata_list)):
                chunk_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

                # Проверка на дубликат
                cur.execute("""
                    SELECT id FROM document_chunks 
                    WHERE chunk_hash = %s AND embedding_model = %s
                """, (chunk_hash, embedding_model))

                if cur.fetchone():
                    continue

                full_meta = {
                    **meta,
                    "ingestion_date": datetime.now().isoformat(),
                    "filename": filename,
                    "chunk_index": i,
                }

                cur.execute("""
                    INSERT INTO document_chunks 
                    (document_id, filename, chunk_index, chunk_text, chunk_hash, 
                     embedding, embedding_model, metadata, document_type)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s::jsonb, %s)
                """, (
                    document_id,
                    filename,
                    i,
                    text,
                    chunk_hash,
                    "[" + ",".join(map(str, embedding)) + "]",
                    embedding_model,
                    json.dumps(full_meta),
                    document_type
                ))
                new_chunks_count += 1

            conn.commit()

        if new_chunks_count > 0:
            st.success(f"✅ Сохранено {new_chunks_count} новых чанков для файла '{filename}'")
        else:
            st.info(f"ℹ️ Все чанки файла '{filename}' уже существуют в базе.")

        return document_id

    except Exception as e:
        st.error(f"❌ Ошибка при сохранении чанков: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return ""


def log_token_usage(
    total_used: int,
    prompt_used: int,
    completion_used: int,
    precached_prompt_used: int = 0,
    balance_entries: Optional[Dict] = None
) -> None:
    """
    Логирует использование токенов + баланс от GigaChat.
    Теперь баланс сохраняется корректно в колонку balance_entries.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO token_usage_log 
                (total_tokens, prompt_tokens, completion_tokens, 
                 precached_tokens, balance_entries)
                VALUES (%s, %s, %s, %s, %s::jsonb)
            """, (
                total_used,
                prompt_used,
                completion_used,
                precached_prompt_used,
                json.dumps(balance_entries) if balance_entries else None
            ))
        conn.commit()

    except Exception as e:
        st.warning(f"⚠️ Не удалось записать лог токенов: {e}")


# ====================== Экспортируемые функции ======================
__all__ = [
    "get_db_connection",
    "init_vector_db",
    "save_chunks",
    "log_token_usage"
]