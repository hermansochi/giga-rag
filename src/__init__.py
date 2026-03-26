# src/__init__.py
"""
Основной пакет src — содержит конфигурацию, базу данных и логику RAG.
"""

from .config import settings
from .database import (
    get_db_connection,
    init_vector_db,
    save_chunks,
    log_token_usage
)
from .gigachat import (
    generate_with_gigachat,
    get_gigachat_client,
    get_available_models,
    display_models
)

__all__ = [
    "settings",
    "get_db_connection",
    "init_vector_db",
    "save_chunks",
    "log_token_usage",
    "generate_with_gigachat",
    "get_gigachat_client",
    "get_available_models",
    "display_models",
]