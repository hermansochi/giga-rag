"""
src/rag/reranker.py

Модуль реранкинга чанков.
Поддерживает LLM-реранкинг и Cross-Encoder (сейчас используется tiny версия).
"""

import streamlit as st
from typing import List, Tuple, Optional

from ..config import settings


def get_gigachat_client():
    """Заглушка для избежания циклического импорта."""
    from ..gigachat import get_gigachat_client as real_get_client
    return real_get_client()


def rerank_chunks(
    query: str,
    chunks: List[Tuple],
    reranker_type: Optional[str] = None,
    top_n: Optional[int] = None
) -> List[Tuple]:
    """
    Универсальная функция реранкинга.
    """
    if not chunks:
        return []

    if top_n is None:
        top_n = settings.RERANK_TOP_N

    use_type = reranker_type or settings.RERANKER_TYPE

    if use_type == "cross_encoder":
        return _rerank_cross_encoder(query, chunks, top_n)
    elif use_type == "llm":
        return _rerank_with_llm(query, chunks, top_n)
    else:
        # "none" или неизвестный тип — возвращаем как есть
        return chunks[:top_n]


def _rerank_cross_encoder(query: str, chunks: List[Tuple], top_n: int) -> List[Tuple]:
    """Реранкинг с помощью Cross-Encoder (tiny версия)."""
    try:
        from sentence_transformers import CrossEncoder
        import torch

        model_name = settings.CROSS_ENCODER_MODEL  # сейчас BAAI/bge-reranker-tiny

        # Определяем устройство
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"🔄 Cross-Encoder: {model_name} на {device} (tiny)")

        model = CrossEncoder(model_name, device=device)

        # Подготавливаем пары (query, chunk_text)
        pairs = [[query, chunk[0]] for chunk in chunks]   # chunk[0] = chunk_text

        # Получаем scores
        scores = model.predict(pairs, show_progress_bar=False)

        # Сортируем по убыванию score
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        return [item[0] for item in ranked[:top_n]]

    except ImportError:
        st.warning("⚠️ Библиотека sentence-transformers не установлена. Используем оригинальный порядок.")
        return chunks[:top_n]
    except Exception as e:
        st.warning(f"⚠️ Ошибка Cross-Encoder реранкинга: {e}")
        return chunks[:top_n]


def _rerank_with_llm(query: str, chunks: List[Tuple], top_n: int) -> List[Tuple]:
    """LLM-реранкинг через GigaChat (оставляем как было)."""
    if len(chunks) <= top_n:
        return chunks

    # ... твой текущий код LLM-реранкинга ...
    # (пока оставляем заглушкой, если не был реализован)
    st.info("🔄 LLM-реранкинг пока не реализован. Возвращаем топ чанков.")
    return chunks[:top_n]