"""
src/rag/reranker.py

Модуль реранкинга чанков для RAG.
Поддерживает cross_encoder (из /app/model_data) и llm (через GigaChat).
"""

from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import re

from ..config import settings


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
) -> List[Tuple[str, Dict[str, Any]]]:
    """Главная функция реранкинга.
    
    Args:
        query: вопрос пользователя
        chunks: список (chunk_text, metadata)
        reranker_type: "llm", "cross_encoder" или None
        top_n: сколько лучших чанков вернуть
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
        st.info("ℹ️ Реранкинг отключён (none). Возвращаем топ по векторному поиску.")
        return chunks[:top_n]


def _rerank_cross_encoder(
    query: str, 
    chunks: List[Tuple[str, Dict[str, Any]]], 
    top_n: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """Реранкинг через Cross-Encoder."""
    model = _get_cross_encoder_model()
    if model is None:
        st.warning("⚠️ Cross-Encoder недоступен.")
        return chunks[:top_n]

    try:
        pairs = [[query, chunk[0]] for chunk in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:top_n]]

    except Exception as e:
        st.warning(f"⚠️ Ошибка Cross-Encoder: {e}")
        return chunks[:top_n]


def _rerank_with_llm(
    query: str, 
    chunks: List[Tuple[str, Dict[str, Any]]], 
    top_n: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """LLM-реранкинг через GigaChat.
    
    Теперь правильно импортируем Chat и Messages внутри функции.
    """
    if len(chunks) <= top_n:
        return chunks

    try:
        client = get_gigachat_client()

        # === Формируем промпт для реранкинга ===
        chunk_list = []
        for i, (text, meta) in enumerate(chunks, 1):
            filename = meta.get("filename", "неизвестный_файл")
            chunk_list.append(f"Чанк №{i} (файл: {filename})\n{text[:700]}...")

        prompt = f"""Ты — эксперт по оценке релевантности.
Вопрос пользователя: "{query}"

Ниже {len(chunks)} фрагментов из документов.
Верни только JSON-массив с номерами чанков в порядке убывания релевантности.

Пример: [3, 1, 5, 2, 4]

Фрагменты:

""" + "\n\n".join(chunk_list)

        st.info(f"🔄 LLM-реранкинг: оцениваю {len(chunks)} чанков через GigaChat...")

        # === ИМПОРТ ЗДЕСЬ — это решает ошибку "name 'Chat' is not defined" ===
        from gigachat.models import Chat, Messages, MessagesRole

        resp = client.chat(
            Chat(
                messages=[Messages(role=MessagesRole.USER, content=prompt)],
                model=settings.GIGACHAT_MODEL,
                temperature=0.1,
            )
        )

        answer = resp.choices[0].message.content.strip()

        # Извлекаем номера чанков из ответа модели
        indices = [int(x) - 1 for x in re.findall(r'\d+', answer) if x.isdigit()]

        valid_indices = [i for i in indices if 0 <= i < len(chunks)]
        
        seen = set()
        final_order = []
        for idx in valid_indices:
            if idx not in seen:
                final_order.append(idx)
                seen.add(idx)

        if not final_order:
            final_order = list(range(len(chunks)))

        return [chunks[i] for i in final_order[:top_n]]

    except Exception as e:
        st.warning(f"⚠️ Ошибка LLM-реранкинга: {type(e).__name__}: {e}")
        return chunks[:top_n]