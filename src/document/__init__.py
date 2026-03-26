# src/document/__init__.py
"""
Пакет для работы с документами: парсинг и подготовка чанков.
"""

from .parser import parse_document, get_supported_extensions
from .chunker import smart_chunk

__all__ = ["parse_document", "get_supported_extensions", "smart_chunk"]