"""
src/rag/reranker.py

Модуль реранкинга чанков с поддержкой bm25 и hybrid.
"""

from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import re

from src.config import settings
from src.models import DocumentChunk, RerankCandidate, RerankedResult


def get_gigachat_client():
    """Заглушка для избежания циклического импорта."""
    from src.gigachat import get_gigachat_client as real_get_client
    return real_get_client()


def rerank_chunks(
    query: str,
    chunks: List[Tuple[str, Dict[str, Any]]],
    reranker_type: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[RerankedResult]:
    """Главная функция реранкинга."""
    if not chunks:
        return []

    if top_n is None:
        top_n = settings.RERANK_TOP_N

    use_type = reranker_type or "none"

    # Преобразуем в кандидаты
    candidates = [RerankCandidate(text=text, metadata=meta) for text, meta in chunks]

    if use_type == "cross_encoder":
        return _rerank_cross_encoder(query, candidates, top_n)
    elif use_type == "llm":
        return _rerank_with_llm(query, candidates, top_n)
    elif use_type == "bm25":
        return _rerank_bm25(query, candidates, top_n)
    elif use_type == "hybrid":
        return _rerank_hybrid(query, candidates, top_n)
    else:
        # none — без реранкинга
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]


# ====================== Cross-Encoder ======================
def _rerank_cross_encoder(query: str, candidates: List[RerankCandidate], top_n: int) -> List[RerankedResult]:
    """Cross-Encoder реранкинг."""
    model = _get_cross_encoder_model()
    if model is None:
        st.warning("⚠️ Cross-Encoder недоступен.")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]

    try:
        pairs = [[query, c.text] for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            RerankedResult(
                text=item[0].text,
                metadata=item[0].metadata,
                rerank_score=float(item[1])
            )
            for item in ranked[:top_n]
        ]
    except Exception as e:
        st.warning(f"⚠️ Ошибка Cross-Encoder: {e}")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]


def _get_cross_encoder_model():
    """Загружает и кэширует Cross-Encoder модель."""
    if "cross_encoder_model" not in st.session_state:
        try:
            from sentence_transformers import CrossEncoder
            import torch

            model_path = "/app/model_data"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            st.info(f"🔄 Загружаю Cross-Encoder из {model_path} на {device}")
            st.session_state.cross_encoder_model = CrossEncoder(model_path, device=device)
        except ImportError:
            st.warning("⚠️ sentence-transformers не установлен.")
            st.session_state.cross_encoder_model = None
        except Exception as e:
            st.error(f"❌ Не удалось загрузить Cross-Encoder: {e}")
            st.session_state.cross_encoder_model = None

    return st.session_state.cross_encoder_model


# ====================== LLM ======================
def _rerank_with_llm(query: str, candidates: List[RerankCandidate], top_n: int) -> List[RerankedResult]:
    """LLM-реранкинг через GigaChat."""
    if len(candidates) <= top_n:
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates]

    try:
        client = get_gigachat_client()

        chunk_list = []
        for i, c in enumerate(candidates, 1):
            filename = c.metadata.get("filename", "неизвестный_файл")
            chunk_list.append(f"Чанк №{i} (файл: {filename})\n{c.text[:700]}...")

        prompt = f"""Ты — эксперт по оценке релевантности.
Вопрос пользователя: "{query}"

Ниже {len(candidates)} фрагментов. Верни только JSON-массив номеров чанков по убыванию релевантности.

Пример: [3, 1, 5, 2, 4]

Фрагменты:

""" + "\n\n".join(chunk_list)

        st.info(f"🔄 LLM-реранкинг: оцениваю {len(candidates)} чанков...")

        from gigachat.models import Chat, Messages, MessagesRole

        resp = client.chat(
            Chat(
                messages=[Messages(role=MessagesRole.USER, content=prompt)],
                model=settings.GIGACHAT_MODEL,
                temperature=0.1,
            )
        )

        answer = resp.choices[0].message.content.strip()

        indices = [int(x) - 1 for x in re.findall(r'\d+', answer) if x.isdigit()]
        valid_indices = [i for i in indices if 0 <= i < len(candidates)]

        seen = set()
        final_order = []
        for idx in valid_indices:
            if idx not in seen:
                final_order.append(idx)
                seen.add(idx)

        if not final_order:
            final_order = list(range(len(candidates)))

        return [RerankedResult(text=candidates[i].text, metadata=candidates[i].metadata) 
                for i in final_order[:top_n]]

    except Exception as e:
        st.warning(f"⚠️ Ошибка LLM-реранкинга: {e}")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]


# ====================== BM25 ======================
def _rerank_bm25(query: str, candidates: List[RerankCandidate], top_n: int) -> List[RerankedResult]:
    """Простой BM25 реранкинг."""
    try:
        from .bm25 import bm25_search

        # Правильное создание DocumentChunk без from_text
        doc_chunks = []
        for c in candidates:
            doc_chunks.append(DocumentChunk(
                chunk_text=c.text,                    # используем chunk_text
                filename=c.metadata.get("filename", "unknown"),
                chunk_index=c.metadata.get("chunk_index", 0),
                distance=c.metadata.get("distance", 0.0),
                metadata=c.metadata
            ))

        bm25_results = bm25_search(query, doc_chunks, top_k=top_n)

        return [
            RerankedResult(
                text=chunk.chunk_text,
                metadata={
                    "filename": chunk.filename,
                    "chunk_index": chunk.chunk_index,
                    "distance": getattr(chunk, 'distance', 0.0)
                }
            ) for chunk in bm25_results
        ]

    except Exception as e:
        st.warning(f"⚠️ Ошибка BM25: {e}")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]

# ====================== Hybrid ======================
def _rerank_hybrid(query: str, candidates: List[RerankCandidate], top_n: int) -> List[RerankedResult]:
    """Гибридный поиск: Vector + BM25 + RRF"""
    from .bm25 import bm25_search
    from src.gigachat import find_relevant_chunks

    try:
        vector_chunks = find_relevant_chunks(query, top_k=50)
        bm25_chunks = bm25_search(query, vector_chunks, top_k=50)

        return _rrf_fusion(vector_chunks, bm25_chunks, top_n)

    except Exception as e:
        st.warning(f"⚠️ Ошибка гибридного поиска: {e}")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]


def _rrf_fusion(vector_chunks: List[DocumentChunk], bm25_chunks: List[DocumentChunk], top_n: int) -> List[RerankedResult]:
    """Reciprocal Rank Fusion"""
    from collections import defaultdict

    scores = defaultdict(float)
    k = 60

    for rank, chunk in enumerate(vector_chunks):
        key = (chunk.filename, chunk.chunk_index)
        scores[key] += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(bm25_chunks):
        key = (chunk.filename, chunk.chunk_index)
        scores[key] += 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    seen = set()

    for (filename, chunk_index), score in sorted_items[:top_n]:
        if (filename, chunk_index) in seen:
            continue
        seen.add((filename, chunk_index))

        for chunk in vector_chunks + bm25_chunks:
            if chunk.filename == filename and chunk.chunk_index == chunk_index:
                result.append(RerankedResult(
                    text=chunk.chunk_text,
                    metadata={
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "distance": getattr(chunk, 'distance', 0.0)
                    },
                    rerank_score=round(score, 4)
                ))
                break

    return result
