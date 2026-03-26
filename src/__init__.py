"""
src/__init__.py

Главный инициализатор пакета src.
Здесь мы собираем всё самое нужное из модулей, чтобы удобно импортировать в других файлах.
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
    get_reranker_options,      # ← новая функция
)

# Что доступно при импорте from src import ...
__all__ = [
    "settings",
    "get_db_connection",
    "init_vector_db",
    "save_chunks",
    "log_token_usage",
    "generate_with_gigachat",
    "get_gigachat_client",
    "get_available_models",
    "get_reranker_options",
]