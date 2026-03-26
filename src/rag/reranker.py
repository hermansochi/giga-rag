# src/rag/reranker.py
"""
Модуль реранкинга чанков.
Поддерживает два режима: llm и cross_encoder.
Можно переключать как через .env, так и через UI.
"""

import streamlit as st
from typing import List, Tuple, Optional

from ..config import settings
from ..config import settings

# Импортируем необходимые классы из gigachat.models
from gigachat.models import Chat, Messages, MessagesRole

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

    Args:
        query: вопрос пользователя
        chunks: список кандидатов из векторного поиска
        reranker_type: "llm" или "cross_encoder". Если None — берётся из настроек.
        top_n: сколько чанков оставить после реранкинга (по умолчанию из настроек)

    Returns:
        Отсортированный список чанков (самые релевантные первые)
    """
    if not chunks:
        return []

    if top_n is None:
        top_n = settings.RERANK_TOP_N

    # Определяем, какой реранкер использовать (UI имеет приоритет)
    use_type = reranker_type or settings.RERANKER_TYPE

    if use_type == "cross_encoder":
        return _rerank_cross_encoder(query, chunks, top_n)
    else:
        # По умолчанию — LLM-реранкинг через GigaChat
        return _rerank_with_llm(query, chunks, top_n)


def _rerank_cross_encoder(query: str, chunks: List[Tuple], top_n: int) -> List[Tuple]:
    """Реранкинг с помощью cross-encoder модели (быстрый и точный)."""
    try:
        from sentence_transformers import CrossEncoder
        import torch

        def load_cross_encoder():
            model_name = settings.CROSS_ENCODER_MODEL
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"🔄 Загружен cross-encoder: {model_name} на {device}")
            return CrossEncoder(model_name, device=device)

        model = load_cross_encoder()

        pairs = [[query, chunk[0]] for chunk in chunks]          # chunk[0] = chunk_text
        scores = model.predict(pairs, show_progress_bar=False)

        # Сортируем по убыванию score
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:top_n]]

    except ImportError:
        st.warning("⚠️ sentence-transformers не установлен. Использую LLM-реранкинг.")
        return chunks[:top_n]
    except Exception as e:
        st.warning(f"⚠️ Ошибка cross-encoder реранкинга: {e}")
        return chunks[:top_n]


def _rerank_with_llm(query: str, chunks: List[Tuple], top_n: int) -> List[Tuple]:
    """Реранкинг через сам GigaChat (не требует дополнительных библиотек)."""
    if len(chunks) <= top_n:
        return chunks

    doc_list = []
    for i, (text, filename, idx, dist) in enumerate(chunks):
        short_text = text[:1500] + "..." if len(text) > 1500 else text
        # Преобразуем dist в float, если это строка
        try:
            dist_value = float(dist)
        except (ValueError, TypeError):
            dist_value = 0.0
        doc_list.append(
            f"Документ [{i}] | Файл: {filename} | Чанк: {idx} | dist: {dist_value:.4f}\n{short_text}"
        )

    prompt = (
        "Ты — эксперт по оценке релевантности документов. "
        "Оцени каждый документ по шкале от 0 до 10, насколько он полезен для ответа на вопрос.\n"
        "Ответь **только** JSON-массивом:\n"
        '[{"index": 0, "score": 9.5, "reason": "кратко почему"}, ...]\n\n'
        f"Вопрос: {query}\n\nДокументы:\n" + "\n\n".join(doc_list) + "\n\nJSON:"
    )

    client = get_gigachat_client()

    try:
        response = client.chat(
            Chat(
                messages=[Messages(role=MessagesRole.USER, content=prompt)],
                model=settings.GIGACHAT_MODEL,
                temperature=0.0,
            )
        )
        content = response.choices[0].message.content.strip()

        # Извлекаем JSON
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if json_match:
            scores_data = json.loads(json_match.group(0))
            sorted_data = sorted(scores_data, key=lambda x: x.get("score", 0), reverse=True)
            return [chunks[item["index"]] for item in sorted_data[:top_n]]

    except Exception as e:
        st.warning(f"⚠️ Ошибка LLM-реранкинга: {e}")

    # Fallback
    return chunks[:top_n]