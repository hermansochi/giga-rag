# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    # ==================== GigaChat ====================
    GIGACHAT_API_KEY: str                    # Обязательное поле, берётся только из .env
    GIGACHAT_SCOPE: str = "GIGACHAT_API_B2B"
    GIGACHAT_MODEL: str = "GigaChat-2-Max"
    EMBEDDING_MODEL: str = "GigaEmbeddings-3B-2025-09"
    EMBEDDING_DIM: int = 2048

    # ==================== RAG ====================
    RERANKER_TYPE: Literal["llm", "cross_encoder"] = "llm"
    RERANK_CANDIDATES: int = 15
    RERANK_TOP_N: int = 5
    CROSS_ENCODER_MODEL: str = "DiTy/cross-encoder-russian-msmarco"

    # ==================== PostgreSQL ====================
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "app"
    POSTGRES_USER: str = "app"
    POSTGRES_PASSWORD: str                     # Обязательное, без значения по умолчания

    # ==================== MinIO ====================
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "documents"
    MINIO_SECURE: bool = False

    # ==================== GigaChat ====================
    BASE_SYSTEM_PROMT: str = "Ты дружелюбный и полезный помощник."
    RAG_SYSTEM_PROMT: str = "Ты — точный помощник. Отвечай строго по документам."
    RAG_PROMT_SUFFIX: str = "Ответь максимально полезно"
    BASE_TEMPERATURE: float = 0.7
    RAG_TEMPERATURE: float = 0.3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )


# Глобальный экземпляр
settings = Settings()