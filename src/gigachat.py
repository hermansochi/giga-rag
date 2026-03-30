"""
src/gigachat.py

Модуль для работы с GigaChat.
Исправлено сохранение баланса: объект Balance преобразуется в dict.
"""

from typing import List, Optional, Tuple, Dict, Any
import time
import html

import streamlit as st
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from src.config import settings
from src.database import get_db_connection, log_token_usage


def get_gigachat_client() -> GigaChat:
    if "gigachat_client" not in st.session_state:
        try:
            st.session_state.gigachat_client = GigaChat(
                credentials=settings.GIGACHAT_API_KEY.strip(),
                scope=settings.GIGACHAT_SCOPE,
                verify_ssl_certs=False,
                timeout=90,
            )
        except Exception as e:
            st.error(f"❌ Не удалось создать клиент GigaChat: {e}")
            st.stop()
    return st.session_state.gigachat_client


def get_available_models() -> List[str]:
    try:
        client = get_gigachat_client()
        response = client.get_models()
        if response and response.data:
            return sorted([getattr(m, 'id_', getattr(m, 'id', str(m))) for m in response.data])
        return [settings.GIGACHAT_MODEL]
    except:
        return [settings.GIGACHAT_MODEL]


def get_reranker_options() -> dict:
    return {
        "none": "Без реранкера (быстрее)",
        "llm": "LLM-реранкинг (через GigaChat)",
        "cross_encoder": "Cross-Encoder"
    }


def find_relevant_chunks(query: str, top_k: int = 15) -> List:
    client = get_gigachat_client()
    try:
        emb_response = client.embeddings(model=settings.EMBEDDING_MODEL, texts=[query])
        query_embedding = emb_response.data[0].embedding

        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_text, filename, chunk_index, embedding <-> %s::VECTOR AS distance
                FROM document_chunks
                ORDER BY distance
                LIMIT %s
            """, (str(query_embedding), top_k))
            return cur.fetchall()
    except Exception as e:
        st.error(f"❌ Ошибка векторного поиска: {e}")
        return []


def get_balance_info(client: GigaChat) -> Optional[Dict]:
    """Получает баланс и преобразует его в обычный словарь."""
    try:
        balance_obj = None

        # Пробуем разные способы получения баланса
        if hasattr(client, 'get_balance'):
            balance_obj = client.get_balance()
        elif hasattr(client, 'balance'):
            balance_obj = client.balance
        elif hasattr(client, 'get_account_balance'):
            balance_obj = client.get_account_balance()

        if balance_obj is None:
            return None

        # Преобразуем объект Balance в словарь
        if hasattr(balance_obj, 'model_dump'):
            # Современный pydantic v2 стиль
            return balance_obj.model_dump()
        elif hasattr(balance_obj, 'dict'):
            # Старый pydantic v1 стиль
            return balance_obj.dict()
        else:
            # Если ничего не помогло — пытаемся через __dict__
            return vars(balance_obj) if hasattr(balance_obj, '__dict__') else str(balance_obj)

    except Exception as e:
        st.warning(f"Не удалось получить баланс: {e}")
        return None


def _get_simple_response(client: GigaChat, prompt: str, model_name: str) -> Tuple[str, int, int]:
    try:
        messages = [
            Messages(role=MessagesRole.SYSTEM, content=st.session_state.get("custom_base_prompt", settings.BASE_SYSTEM_PROMT)),
            Messages(role=MessagesRole.USER, content=prompt)
        ]

        resp = client.chat(Chat(messages=messages, model=model_name))
        answer = resp.choices[0].message.content.strip()

        # Получаем баланс отдельным методом
        balance_info = get_balance_info(client)

        log_token_usage(
            total_used=getattr(resp.usage, 'total_tokens', 0),
            prompt_used=getattr(resp.usage, 'prompt_tokens', 0),
            completion_used=getattr(resp.usage, 'completion_tokens', 0),
            precached_prompt_used=getattr(resp.usage, 'precached_prompt_tokens', 0),
            balance_entries=balance_info
        )

        return answer, getattr(resp.usage, 'prompt_tokens', 0), getattr(resp.usage, 'completion_tokens', 0)

    except Exception as e:
        st.warning(f"⚠️ Ошибка простого запроса: {e}")
        return "Не удалось сгенерировать ответ 😔", 0, 0


def _get_rag_response(client: GigaChat, prompt: str, relevant_chunks: List, model_name: str) -> Tuple[str, int, int]:
    if not relevant_chunks:
        return "В базе документов пока нет информации по этому вопросу.", 0, 0

    context_parts = []
    for i, row in enumerate(relevant_chunks, 1):
        text = str(row.get("chunk_text", "")).strip()
        filename = str(row.get("filename", "неизвестный_файл"))
        chunk_idx = str(row.get("chunk_index", "?"))
        distance = float(row.get("distance", 0.0))
        context_parts.append(f"--- Источник [{i}] | {filename} | чанк {chunk_idx} | dist {distance:.4f} ---\n{text}")

    system_prompt = st.session_state.get("custom_rag_prompt", settings.RAG_SYSTEM_PROMT)

    rag_suffix = st.session_state.get("custom_rag_suffix", settings.RAG_PROMT_SUFFIX)
    user_msg = "\n\n".join(context_parts) + f"\n\nВопрос: {prompt}\n{rag_suffix}:"

    try:
        resp = client.chat(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_msg)
                ],
                model=model_name,
                temperature=st.session_state.get("custom_rag_temperature", settings.RAG_TEMPERATURE),
            )
        )

        answer = resp.choices[0].message.content.strip()

        # Получаем баланс отдельным методом
        balance_info = get_balance_info(client)

        log_token_usage(
            total_used=getattr(resp.usage, 'total_tokens', 0),
            prompt_used=getattr(resp.usage, 'prompt_tokens', 0),
            completion_used=getattr(resp.usage, 'completion_tokens', 0),
            precached_prompt_used=getattr(resp.usage, 'precached_prompt_tokens', 0),
            balance_entries=balance_info
        )

        return answer, getattr(resp.usage, 'prompt_tokens', 0), getattr(resp.usage, 'completion_tokens', 0)

    except Exception as e:
        st.warning(f"⚠️ Ошибка RAG-запроса: {e}")
        return "Не удалось получить ответ на основе документов 😔", 0, 0


def generate_with_gigachat(
    prompt: str,
    model_name: Optional[str] = None,
    use_rag: bool = False,
    reranker_type: Optional[str] = None,
) -> str:
    """
    Генерирует ответ с поддержкой RAG и отображает использованный контекст.
    """
    model_name = model_name or settings.GIGACHAT_MODEL
    client = get_gigachat_client()
    start_time = time.time()

    retrieved_chunks_log = []
    full_context = None

    if not use_rag:
        answer, prompt_tokens, completion_tokens = _get_simple_response(client, prompt, model_name)
        sources_html = ""
        context_details = ""
    else:
        relevant_chunks = find_relevant_chunks(prompt)

        # Лог для БД
        retrieved_chunks_log = [
            {
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "distance": float(row["distance"]),
                "text": row["chunk_text"]
            }
            for row in relevant_chunks
        ]

        # Формируем контекст (передаётся в модель)
        context_parts = []
        for i, row in enumerate(relevant_chunks, 1):
            text = str(row.get("chunk_text", "")).strip()
            filename = str(row.get("filename", "неизвестный_файл"))
            chunk_idx = str(row.get("chunk_index", "?"))
            distance = float(row.get("distance", 0.0))
            context_parts.append(f"--- Источник [{i}] | {filename} | чанк {chunk_idx} | dist {distance:.4f} ---\n{text}")

        rag_suffix = st.session_state.get("custom_rag_suffix", settings.RAG_PROMT_SUFFIX)
        full_context = "\n\n".join(context_parts) + f"\n\nВопрос: {prompt}\n{rag_suffix}:"

        answer, prompt_tokens, completion_tokens = _get_rag_response(client, prompt, relevant_chunks, model_name)

        # 🔽 Блок с источниками
        sources_parts = []
        for row in relevant_chunks:
            filename = row["filename"]
            distance = float(row["distance"])
            sources_parts.append(f"📄 `{filename}` (расстояние: {distance:.4f})")

        sources_html = ""
        if sources_parts:
            sources_html = f"""
<div style='font-size:0.85em; color:#555; margin-top:16px; padding:12px; background:#f0f4f8; border-radius:8px; border-left: 4px solid #1f77b4;'>
    <b>📚 Использованные источники:</b><br>""" + \
            "<br>".join(sources_parts) + \
            """
</div>
"""

        # 🔽 Блок с контекстом (скрытый)
        # Ограничиваем длину preview
        preview_context = (full_context[:2000] + "...") if len(full_context) > 2000 else full_context
        context_details = f"""
<details style="margin-top: 16px;">
<summary style="color:#1a73e8; cursor:pointer; font-size:0.9em;">🔍 Показать контекст, переданный в GigaChat</summary>
<div style="margin-top:10px; padding:12px; border:1px solid #ddd; border-radius:8px; background:#f9f9f9; font-family: monospace; font-size:0.85em; white-space: pre-wrap; max-height: 400px; overflow-y: auto;">
{html.escape(preview_context)}
</div>
</details>
"""
    duration = round(time.time() - start_time, 2)
    total_tokens = prompt_tokens + completion_tokens

    metrics_html = f"""
<div style='color:#666; font-size:0.78em; margin-top:14px; padding:10px; background:#f8f9fa; border-radius:8px;'>
    ⏱ Время: <b>{duration} сек</b><br>
    📊 Токены: <b>{total_tokens:,}</b> 
    (prompt: {prompt_tokens:,} | completion: {completion_tokens:,})
</div>
"""

    # 🔽 Логируем в БД
    try:
        from src.database import log_chat_interaction
        log_chat_interaction(
            user_message=prompt,
            assistant_response=answer,
            model_name=model_name,
            use_rag=use_rag,
            reranker_type=reranker_type if use_rag else None,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_time=duration,
            metadata={
                "session_id": st.session_state.get("session_id", "unknown"),
                "page": "chat"
            },
            rag_context=full_context,
            retrieved_chunks=retrieved_chunks_log
        )
    except Exception as e:
        st.warning("⚠️ Не удалось сохранить лог")

    # 🔽 Собираем финальный ответ
    return f"{answer}\n\n{sources_html}\n{context_details}\n{metrics_html}".strip()