"""
src/gigachat.py

Модуль для работы с GigaChat.
Исправлено сохранение баланса: объект Balance преобразуется в dict.
"""

from typing import List, Optional, Tuple, Dict, Any
import time

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
            Messages(role=MessagesRole.SYSTEM, content="Ты дружелюбный и полезный помощник."),
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

    system_prompt = "Ты — точный помощник. Отвечай строго по документам."

    user_msg = "\n\n".join(context_parts) + f"\n\nВопрос: {prompt}\nОтветь максимально полезно:"

    try:
        resp = client.chat(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_msg)
                ],
                model=model_name,
                temperature=0.3,
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
    model_name = model_name or settings.GIGACHAT_MODEL
    client = get_gigachat_client()
    start_time = time.time()

    if not use_rag:
        answer, prompt_tokens, completion_tokens = _get_simple_response(client, prompt, model_name)
    else:
        relevant_chunks = find_relevant_chunks(prompt)
        answer, prompt_tokens, completion_tokens = _get_rag_response(client, prompt, relevant_chunks, model_name)

    duration = round(time.time() - start_time, 2)
    total_tokens = prompt_tokens + completion_tokens

    metrics_html = f"""
<div style='color:#666; font-size:0.78em; margin-top:14px; padding:10px; background:#f8f9fa; border-radius:8px;'>
    ⏱ Время: <b>{duration} сек</b><br>
    📊 Токены: <b>{total_tokens:,}</b> 
    (prompt: {prompt_tokens:,} | completion: {completion_tokens:,})
</div>
"""

    return f"{answer}\n\n{metrics_html}".strip()