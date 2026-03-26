"""
Модуль для работы с GigaChat: эмбеддинги, чат, векторный поиск.
Чистая версия без реранкинга, с фокусом на стабильность и правильную работу.
"""

import time
from typing import List, Tuple, Optional

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import streamlit as st

from .config import settings
from .database import get_db_connection, log_token_usage


def get_gigachat_client():
    """
    Кэшированный клиент GigaChat с правильной передачей токена.
    """
    try:
        # Важно: credentials должен быть именно строкой с токеном, а не путём к файлу
        return GigaChat(
            credentials=settings.GIGACHAT_API_KEY.strip(),
            scope=settings.GIGACHAT_SCOPE,
            verify_ssl_certs=False,
            timeout=60,
        )
    except Exception as e:
        st.error(f"❌ Не удалось инициализировать GigaChat клиент: {e}")
        st.stop()


def get_available_models() -> List[str]:
    """Возвращает список доступных моделей GigaChat."""
    try:
        client = get_gigachat_client()
        response = client.get_models()
        return sorted([model.id for model in response.data]) if response.data else []
    except Exception:
        return []


def display_models(models: List[str], selected_model: str = None):
    """Отображает доступные модели в Streamlit."""
    if not models:
        st.error("❌ Не удалось получить список моделей GigaChat")
        return

    st.write("**Доступные модели GigaChat:**")
    for model in models:
        if model == selected_model:
            st.markdown(f"- `{model}` ✅ (выбрана)")
        else:
            st.markdown(f"- `{model}`")


def find_relevant_chunks(query: str, top_k: int = 15) -> List[Tuple]:
    """Векторный поиск чанков."""
    client = get_gigachat_client()
    try:
        emb_response = client.embeddings(model=settings.EMBEDDING_MODEL, texts=[query])
        query_embedding = emb_response.data[0].embedding

        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_text, filename, chunk_index,
                       embedding <-> %s::VECTOR AS distance
                FROM document_chunks
                ORDER BY distance
                LIMIT %s
            """, (str(query_embedding), top_k))
            return cur.fetchall()
    except Exception as e:
        st.error(f"❌ Ошибка поиска: {e}")
        return []


# ====================== Основная функция ======================
def generate_with_gigachat(
    prompt: str,
    model_name: Optional[str] = None,
    reranker_override: Optional[str] = None,
) -> str:
    """
    Генерирует ответ с RAG (простой векторный поиск, без реранкинга).

    Args:
        prompt: вопрос пользователя
        model_name: модель (по умолчанию из config)
        reranker_override: игнорируется, реранкинг отключён
    """
    model_name = model_name or settings.GIGACHAT_MODEL
    client = get_gigachat_client()

    start_time = time.time()

    # ── 1. Простой ответ без RAG ─────────────────────────────────────
    simple_answer = _get_simple_response(client, prompt, model_name)

    # ── 2. RAG с векторным поиском ────────────────────────────────────
    relevant_chunks = find_relevant_chunks(prompt, top_k=settings.RERANK_CANDIDATES)

    rag_answer = _get_rag_response(client, prompt, relevant_chunks, model_name)
    duration = round(time.time() - start_time, 2)  # время в секундах

    # ── 3. Финальный ответ + метрики ─────────────────────────────────
    final_response = f"""
### 🤖 Ответ без контекста
{simple_answer}

---\n
### 🧠 Ответ с RAG (по документам)
{rag_answer}
"""

    # Формирование списка источников с проверкой на RealDictRow
    if relevant_chunks:
        sources = []
        for i, row in enumerate(relevant_chunks):
            # Проверяем, что row поддерживает индексацию по ключу
            if not hasattr(row, '__getitem__'):
                st.warning(f"Пропущен чанк {i}: объект не поддерживает доступ по ключу (row={row})")
                continue

            # Проверяем наличие обязательных полей
            required_keys = ['filename', 'chunk_index', 'chunk_text']
            if not all(key in row for key in required_keys):
                missing = [key for key in required_keys if key not in row]
                st.warning(f"Пропущен чанк {i}: отсутствуют поля {missing} (row={row})")
                continue

            filename = str(row['filename']) if row['filename'] is not None else "неизвестный_файл"
            chunk_idx = str(row['chunk_index']) if row['chunk_index'] is not None else "?"

            # Обработка расстояния
            if 'distance' in row and row['distance'] not in (None, '', 'None', 'none'):
                try:
                    dist = float(row['distance'])
                    sources.append(f"- `{filename}` (чанк {chunk_idx}, dist={dist:.4f})")
                    continue
                except (ValueError, TypeError):
                    pass  # Переход к упрощённому формату

            sources.append(f"- `{filename}` (чанк {chunk_idx})")

        if sources:
            final_response += f"\n\n📚 **Источники:**\n" + "\n".join(sources[:6])

    # Метрики внизу мелким серым текстом
    global simple_answer_tokens, rag_answer_tokens, prompt_tokens, completion_tokens
    if 'simple_answer_tokens' not in globals():
        simple_answer_tokens = 0
    if 'rag_answer_tokens' not in globals():
        rag_answer_tokens = 0
    if 'prompt_tokens' not in globals():
        prompt_tokens = 0
    if 'completion_tokens' not in globals():
        completion_tokens = 0

    metrics_html = (
        f"<div style='color: #666666; font-size: 0.78em; margin-top: 12px;'>\n"
        f"    ⏱ Время: <b>{duration} сек</b> &nbsp;&nbsp;&nbsp; \n"
        f"    📊 Токены: <b>{simple_answer_tokens + rag_answer_tokens}</b> \n"
        f"    (prompt: {prompt_tokens} | completion: {completion_tokens})\n"
        f"</div>"
    )
    final_response += metrics_html

    return final_response.strip()


# ====================== Вспомогательные функции ======================
def _get_simple_response(client, prompt: str, model_name: str) -> str:
    """Простой запрос без RAG."""
    try:
        messages = [
            Messages(role=MessagesRole.SYSTEM, content="Ты дружелюбный помощник. Отвечай естественно."),
            Messages(role=MessagesRole.USER, content=prompt)
        ]

        resp = client.chat(Chat(messages=messages, model=model_name))
        answer = resp.choices[0].message.content.strip()

        # Логирование токенов
        log_token_usage(
            total_used=resp.usage.total_tokens,
            prompt_used=resp.usage.prompt_tokens,
            completion_used=resp.usage.completion_tokens,
            precached_prompt_used=getattr(resp.usage, 'precached_prompt_tokens', 0),
            balance_entries=None
        )

        return answer

    except Exception as e:
        st.warning(f"⚠️ Ошибка простого запроса: {e}")
        return "Не удалось получить ответ без контекста."


def _get_rag_response(client, prompt: str, relevant_chunks: List[Tuple], model_name: str) -> str:
    """RAG-ответ с правильным логированием токенов и метриками."""
    if not relevant_chunks:
        return "В предоставленных материалах информация по этому вопросу не найдена."

    context_parts = []
    for i, row in enumerate(relevant_chunks, 1):
        if not hasattr(row, '__getitem__'):
            continue
        text = str(row['chunk_text']).strip() if row['chunk_text'] else ""
        filename = str(row['filename']) if row['filename'] else "неизвестный_файл"
        idx = str(row['chunk_index']) if row['chunk_index'] is not None else "?"

        dist_value = 0.0
        if 'distance' in row and row['distance'] not in (None, '', 'None', 'none'):
            try:
                dist_value = float(row['distance'])
            except (ValueError, TypeError):
                dist_value = 0.0

        context_parts.append(
            f"--- Источник [{i}] | {filename} | чанк {idx} | dist {dist_value:.4f} ---\n{text}"
        )

    system_prompt = (
        "Ты — точный помощник. Отвечай строго по документам. "
        "Если есть хоть частичная релевантность — используй её."
    )

    user_msg = "\n\n".join(context_parts) + f"\n\nВопрос: {prompt}\nОтветь максимально полезно:"

    try:
        resp = client.chat(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_msg)
                ],
                model=model_name,
                temperature=0.3
            )
        )
        answer = resp.choices[0].message.content.strip()

        log_token_usage(
            total_used=resp.usage.total_tokens,
            prompt_used=resp.usage.prompt_tokens,
            completion_used=resp.usage.completion_tokens,
            precached_prompt_used=getattr(resp.usage, 'precached_prompt_tokens', 0),
            balance_entries=None
        )

        # Обновляем глобальные переменные для отображения метрик
        global rag_answer_tokens, prompt_tokens, completion_tokens
        rag_answer_tokens = resp.usage.total_tokens
        prompt_tokens = resp.usage.prompt_tokens
        completion_tokens = resp.usage.completion_tokens

        return answer

    except Exception as e:
        st.warning(f"⚠️ Ошибка RAG-запроса: {e}")
        return "Не удалось получить ответ с документами."