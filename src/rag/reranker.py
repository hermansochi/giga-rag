"""
src/rag/reranker.py

Модуль реранкинга чанков.
Полностью переведён на DTO (RerankCandidate → RerankedResult).
"""

from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import re

from ..config import settings
from src.models import RerankCandidate, RerankedResult


def get_gigachat_client():
    """Заглушка для избежания циклического импорта."""
    from ..gigachat import get_gigachat_client as real_get_client
    return real_get_client()


def _get_cross_encoder_model():
    """Загружает и кэширует Cross-Encoder модель из /app/model_data (один раз за сессию)."""
    if "cross_encoder_model" not in st.session_state:
        try:
            from sentence_transformers import CrossEncoder
            import torch

            model_path = "/app/model_data"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            st.info(f"🔄 Загружаю Cross-Encoder из {model_path} на {device}")
            st.session_state.cross_encoder_model = CrossEncoder(model_path, device=device)
        except ImportError:
            st.warning("⚠️ sentence-transformers не установлен. Cross-Encoder отключён.")
            st.session_state.cross_encoder_model = None
        except Exception as e:
            st.error(f"❌ Не удалось загрузить Cross-Encoder: {e}")
            st.session_state.cross_encoder_model = None

    return st.session_state.cross_encoder_model


def rerank_chunks(
    query: str,
    chunks: List[Tuple[str, Dict[str, Any]]],
    reranker_type: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[RerankedResult]:
    """Главная функция реранкинга.
    
    Принимает List[Tuple] для совместимости со старым вызовом из gigachat.py,
    но внутри сразу преобразует в RerankCandidate.
    """
    if not chunks:
        return []

    if top_n is None:
        top_n = settings.RERANK_TOP_N

    use_type = reranker_type or settings.RERANKER_TYPE

    # Преобразуем входные данные в RerankCandidate
    candidates: List[RerankCandidate] = [
        RerankCandidate(text=text, metadata=meta)
        for text, meta in chunks
    ]

    if use_type == "cross_encoder":
        return _rerank_cross_encoder(query, candidates, top_n)
    elif use_type == "llm":
        return _rerank_with_llm(query, candidates, top_n)
    else:
        st.info("ℹ️ Реранкинг отключён (none). Возвращаем топ чанков.")
        return [
            RerankedResult(text=c.text, metadata=c.metadata)
            for c in candidates[:top_n]
        ]


def _rerank_cross_encoder(
    query: str,
    candidates: List[RerankCandidate],
    top_n: int
) -> List[RerankedResult]:
    """Реранкинг через Cross-Encoder модель."""
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
        st.warning(f"⚠️ Ошибка Cross-Encoder реранкинга: {e}")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]


def _rerank_with_llm(
    query: str,
    candidates: List[RerankCandidate],
    top_n: int
) -> List[RerankedResult]:
    """LLM-реранкинг через GigaChat (самый умный вариант)."""
    if len(candidates) <= top_n:
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates]

    try:
        client = get_gigachat_client()

        chunk_list = []
        for i, c in enumerate(candidates, 1):
            filename = c.metadata.get("filename", "неизвестный_файл")
            chunk_list.append(f"Чанк №{i} (файл: {filename})\n{c.text[:700]}...")

        prompt = f"""Ты — эксперт по оценке релевантности текста.
Вопрос пользователя: "{query}"

Ниже {len(candidates)} фрагментов из документов.
Верни только JSON-массив с номерами чанков в порядке убывания релевантности.

Пример ответа: [3, 1, 5, 2, 4]

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

        # Извлекаем номера чанков
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

        return [
            RerankedResult(text=candidates[i].text, metadata=candidates[i].metadata)
            for i in final_order[:top_n]
        ]

    except Exception as e:
        st.warning(f"⚠️ Ошибка LLM-реранкинга: {e}")
        return [RerankedResult(text=c.text, metadata=c.metadata) for c in candidates[:top_n]]
