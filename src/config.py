# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    # ==================== GigaChat ====================
    GIGACHAT_API_KEY: str  # Обязательное поле, берётся только из .env
    GIGACHAT_SCOPE: str = "GIGACHAT_API_B2B"
    GIGACHAT_MODEL: str = "GigaChat-2-Max"
    EMBEDDING_MODEL: str = "GigaEmbeddings-3B-2025-09"
    EMBEDDING_DIM: int = 2048

    # Модели и пути к ним
    CROSS_ENCODER_MODEL: str = "DiTy/cross-encoder-russian-msmarco"
    CROSS_ENCODER_MODEL_PATH: str = "/app/model_data/cross_encoder"

    SPACY_MODEL_NAME: str = "ru_core_news_md"
    SPACY_MODEL_PATH: str = "/app/model_data/spacy/ru_core_news_md"

    # ==================== RAG ====================
    # Все возможные типы реранкинга
    RERANKER_TYPE: Literal[
        "none",           # Без реранкинга
        "llm",            # Через GigaChat
        "cross_encoder",  # Нейросетевой Cross-Encoder
        "bm25",           # Полнотекстовый BM25
        "hybrid"          # Гибридный: Vector + BM25 + RRF
    ] = "none"            # По умолчанию — без реранкинга (безопасно)
    RERANK_CANDIDATES: int = 15
    RERANK_TOP_N: int = 5

    # ==================== PostgreSQL ====================
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "app"
    POSTGRES_USER: str = "app"
    POSTGRES_PASSWORD: str  # Обязательное, без значения по умолчания

    # ==================== MinIO ====================
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "documents"
    MINIO_SECURE: bool = False

    # ==================== GigaChat ====================
    BASE_SYSTEM_PROMT: str = "Ты дружелюбный и полезный помощник."
    RAG_SYSTEM_PROMT: str = "Ответь максимально полезно и информативно на основе предоставленного контекста. Если информации недостаточно — честно скажи об этом."
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
