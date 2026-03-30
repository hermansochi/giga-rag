"""
src/models/__init__.py

Экспортируем все DTO для удобного импорта.
"""

from .dto import (
    DocumentChunk,
    RerankCandidate,
    RerankedResult,
    ChatSource,
    ChatResponse,
    ParsedDocument,
    Chunk, 
)

__all__ = ["DocumentChunk", "RerankCandidate", "RerankedResult", "ChatSource", "ChatResponse", "ParsedDocument", "Chunk"]