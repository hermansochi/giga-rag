# src/rag/__init__.py
"""
Пакет RAG: retrieval, reranking и промпты.
"""

from ..config import settings
from .reranker import rerank_chunks

__all__ = ["rerank_chunks", "settings"]
