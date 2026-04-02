"""
src/rag/bm25.py
Простая и эффективная реализация BM25 для гибридного поиска.
"""

from typing import List
import nltk
from rank_bm25 import BM25Okapi

from src.models import DocumentChunk


def bm25_search(
    query: str, chunks: List[DocumentChunk], top_k: int = 40
) -> List[DocumentChunk]:
    """Выполняет BM25 поиск и возвращает топ-K чанков."""
    if not chunks:
        return []

    try:
        # Скачиваем токенизатор один раз
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        # Подготавливаем данные
        corpus = [chunk.chunk_text.lower().split() for chunk in chunks]
        tokenized_query = query.lower().split()

        # Создаём BM25
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        # Сортируем чанки по score
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        return [chunk for chunk, score in scored[:top_k]]

    except Exception as e:
        print(f"[BM25] Ошибка: {e}")
        # Fallback — возвращаем как есть
        return chunks[:top_k]
