from typing import List, Optional, Tuple, Dict, Any
import time
import html

import streamlit as st
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from src.config import settings
from src.database import get_db_connection, log_token_usage, log_chat_interaction   # ← вот так
from src.models import DocumentChunk, RerankCandidate, RerankedResult

def get_gigachat_client() -> GigaChat:
    """Возвращает (или создаёт) клиент GigaChat. Хранится в session_state."""
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
    """Возвращает список доступных моделей GigaChat."""
    try:
        client = get_gigachat_client()
        response = client.get_models()
        if response and response.data:
            return sorted([getattr(m, 'id_', getattr(m, 'id', str(m))) for m in response.data])
        return [settings.GIGACHAT_MODEL]
    except Exception:
        return [settings.GIGACHAT_MODEL]

def get_reranker_options() -> Dict[str, str]:
    """Возвращает варианты реранкера для UI."""
    return {
        "none": "Без реранкера (быстрее)",
        "llm": "LLM-реранкинг (через GigaChat)",
        "cross_encoder": "Cross-Encoder",
        "bm25": "BM25 (полнотекстовый поиск)",
        "hybrid": "Гибридный поиск (Vector + BM25 + RRF)"
    }

def find_relevant_chunks(query: str, top_k: int = 15) -> List[DocumentChunk]:
    """Выполняет векторный поиск и возвращает список DocumentChunk.
    
    Теперь используем DTO вместо сырых словарей — код стал чище и типобезопаснее.
    """
    client = get_gigachat_client()
    try:
        embedding_model = getattr(settings, "EMBEDDING_MODEL", "Embeddings")
        st.info(f"🔍 Генерирую эмбеддинг для запроса: '{query[:50]}...' (модель: {embedding_model})")

        emb_response = client.embeddings(model=embedding_model, texts=[query])

        if not emb_response or not emb_response.data:
            st.error("❌ Пустой ответ от embeddings API")
            return []

        query_embedding = emb_response.data[0].embedding
        if not query_embedding:
            st.error("❌ В ответе embeddings отсутствует поле 'embedding'")
            return []

        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_text, filename, chunk_index, embedding <-> %s::VECTOR AS distance
                FROM document_chunks
                ORDER BY distance
                LIMIT %s
            """, (str(query_embedding), top_k))
            raw_rows = cur.fetchall()

        if not raw_rows:
            st.info("ℹ️ По запросу ничего не найдено в базе документов.")
            return []

        # ← Главное изменение: используем DTO
        return [DocumentChunk.from_db_row(row) for row in raw_rows]

    except Exception as e:
        st.error(f"❌ Ошибка векторного поиска: {type(e).__name__}: {e}")
        st.error(f"Traceback:\n{traceback.format_exc()[:800]}...")
        return []

def get_balance_info(client: GigaChat) -> Optional[Dict[str, Any]]:
    """Получает баланс и преобразует в словарь."""
    try:
        balance_obj = None
        if hasattr(client, 'get_balance'):
            balance_obj = client.get_balance()
        elif hasattr(client, 'balance'):
            balance_obj = client.balance
        elif hasattr(client, 'get_account_balance'):
            balance_obj = client.get_account_balance()

        if balance_obj is None:
            return None

        if hasattr(balance_obj, 'model_dump'):
            return balance_obj.model_dump()
        elif hasattr(balance_obj, 'dict'):
            return balance_obj.dict()
        return vars(balance_obj) if hasattr(balance_obj, '__dict__') else str(balance_obj)

    except Exception as e:
        st.warning(f"Не удалось получить баланс: {e}")
        return None

def _get_simple_response(client: GigaChat, prompt: str, model_name: str) -> Tuple[str, int, int]:
    """Простой запрос без RAG."""
    try:
        messages = [
            Messages(role=MessagesRole.SYSTEM, content=st.session_state.get("custom_base_prompt", settings.BASE_SYSTEM_PROMT)),
            Messages(role=MessagesRole.USER, content=prompt)
        ]

        resp = client.chat(Chat(messages=messages, model=model_name))
        answer = resp.choices[0].message.content.strip()

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

def _get_rag_response(client: GigaChat, prompt: str, relevant_chunks: List[DocumentChunk], model_name: str) -> Tuple[str, int, int]:
    """Генерация ответа с использованием RAG-контекста.
    Использует DocumentChunk для формирования промпта.
    """
    if not relevant_chunks:
        return "В базе документов пока нет информации по этому вопросу.", 0, 0

    # Формируем контекст из DTO
    context_parts = [
        f"--- Источник [{i}] | {chunk.filename} | чанк {chunk.chunk_index} | dist {chunk.distance:.4f} ---\n"
        f"{chunk.full_preview}"
        for i, chunk in enumerate(relevant_chunks, 1)
    ]

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
) -> tuple[str, Dict[str, Any]]:
    """
    Главная функция генерации ответа.
    Возвращает: (response_html: str, metrics: dict)
    """
    model_name = model_name or settings.GIGACHAT_MODEL
    client = get_gigachat_client()
    start_time = time.time()

    retrieved_chunks_log: List[Dict[str, Any]] = []
    full_context: Optional[str] = None
    relevant_chunks: List[DocumentChunk] = []

    if not use_rag:
        answer, prompt_tokens, completion_tokens = _get_simple_response(client, prompt, model_name)
        sources_html = ""
        context_details = ""
    else:
        # 1. Векторный поиск
        relevant_chunks_raw: List[DocumentChunk] = find_relevant_chunks(prompt, settings.RERANK_CANDIDATES)

        # 2. Подготовка кандидатов
        candidate_chunks: List[RerankCandidate] = [
            RerankCandidate.from_document_chunk(chunk) for chunk in relevant_chunks_raw
        ]

        # 3. Реранкинг
        if reranker_type and reranker_type != "none":
            from src.rag.reranker import rerank_chunks
            st.info(f"🔄 Применяю реранкинг: {reranker_type}")

            ranked_results: List[RerankedResult] = rerank_chunks(
                query=prompt,
                chunks=[(c.text, c.metadata) for c in candidate_chunks],
                reranker_type=reranker_type,
                top_n=settings.RERANK_TOP_N
            )

            relevant_chunks = [
                DocumentChunk(
                    chunk_text=result.text,
                    filename=result.metadata.get("filename", "неизвестный"),
                    chunk_index=result.metadata.get("chunk_index", 0),
                    distance=result.metadata.get("distance", 0.0),
                    metadata=result.metadata
                )
                for result in ranked_results
            ]
        else:
            relevant_chunks = relevant_chunks_raw

        # 4. Лог чанков
        retrieved_chunks_log = [
            {
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "distance": chunk.distance,
                "text": chunk.chunk_text[:500]
            }
            for chunk in relevant_chunks
        ]

        # 5. Контекст
        context_parts = [
            f"--- Источник [{i}] | {chunk.filename} | чанк {chunk.chunk_index} | dist {chunk.distance:.4f} ---\n"
            f"{chunk.full_preview}"
            for i, chunk in enumerate(relevant_chunks, 1)
        ]

        rag_suffix = st.session_state.get("custom_rag_suffix", settings.RAG_PROMT_SUFFIX)
        full_context = "\n\n".join(context_parts) + f"\n\nВопрос: {prompt}\n{rag_suffix}:"

        # 6. Генерация ответа
        answer, prompt_tokens, completion_tokens = _get_rag_response(client, prompt, relevant_chunks, model_name)

        # 7. HTML-блоки
        sources_html = _build_sources_html(relevant_chunks)
        context_details = _build_context_details(
            full_context=full_context,
            reranked_chunks=relevant_chunks,
            reranker_type=reranker_type
        )

    duration = round(time.time() - start_time, 2)
    total_tokens = prompt_tokens + completion_tokens

    # 8. Метрики HTML
    metrics_html = _build_metrics_html(duration, total_tokens, prompt_tokens, completion_tokens)

    final_response = f"{answer}\n\n{sources_html}\n{context_details}\n{metrics_html}".strip()

    # ====================== ЧИСТЫЕ МЕТРИКИ ======================
    metrics: Dict[str, Any] = {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "response_time": duration,
        "model_name": model_name,
        "use_rag": use_rag,
        "reranker_type": reranker_type if use_rag else None,
    }

    return final_response, metrics

def _build_sources_html(chunks: List[DocumentChunk]) -> str:
    """Expandable блок «Использованные источники» — в едином стиле с контекстом."""
    if not chunks:
        return ""

    sources_parts = []
    for chunk in chunks:
        preview = chunk.short_text.replace('\n', ' ').strip()
        if len(preview) > 280:
            preview = preview[:280] + "..."
        
        sources_parts.append(
            f"<b>📄 {chunk.filename}</b> (dist: {chunk.distance:.4f})<br>"
            f"<small>Чанк {chunk.chunk_index} • {preview}</small>"
        )

    sources_content = "<br><br>".join(sources_parts)

    return f"""
<details style="margin-top: 16px;">
<summary style="color:#1a73e8; cursor:pointer; font-size:0.95em; font-weight:600;">
    📚 Использованные источники ({len(chunks)} шт.)
</summary>
<div style="margin-top:10px; padding:14px; border:1px solid #ddd; border-radius:8px; 
            background:#f9f9f9; font-size:0.87em; line-height:1.5;">
    {sources_content}
</div>
</details>
"""


def _build_context_details(
    full_context: Optional[str],
    reranked_chunks: Optional[List[DocumentChunk]] = None,
    reranker_type: Optional[str] = None
) -> str:
    """Expandable блок контекста в едином стиле."""
    if not full_context:
        return ""

    lines = []

    if reranker_type and reranker_type != "none":
        lines.append(f"<b>🔄 Применён реранкинг:</b> {reranker_type}<br>")

    lines.append("<b>Итоговый запрос, отправленный в GigaChat:</b>")
    lines.append(
        f"<pre style='background:#f8f9fa; padding:12px; border-radius:6px; "
        f"overflow:auto; font-size:0.85em; white-space: pre-wrap; line-height:1.5;'>"
        f"{html.escape(full_context)}</pre>"
    )

    if reranked_chunks:
        lines.append("<hr style='margin:12px 0;'>")
        lines.append("<b>Финальные чанки после реранкинга:</b><br>")
        for i, chunk in enumerate(reranked_chunks, 1):
            lines.append(
                f"• [{i}] <b>{chunk.filename}</b> | чанк {chunk.chunk_index} | "
                f"dist {chunk.distance:.4f}"
            )

    context_html = "\n".join(lines)

    return f"""
<details style="margin-top: 16px;">
<summary style="color:#1a73e8; cursor:pointer; font-size:0.95em; font-weight:600;">
    🔍 Показать контекст, переданный в GigaChat
</summary>
<div style="margin-top:10px; padding:14px; border:1px solid #ddd; border-radius:8px; 
            background:#f9f9f9; font-size:0.87em;">
    {context_html}
</div>
</details>
"""

def _build_metrics_html(duration: float, total_tokens: int, 
                       prompt_tokens: int, completion_tokens: int) -> str:
    """Формирует HTML-блок с метриками (время + токены)."""
    return f"""
<div style='color:#666; font-size:0.78em; margin-top:14px; padding:10px; background:#f8f9fa; border-radius:8px;'>
    ⏱ Время: <b>{duration} сек</b><br>
    📊 Токены: <b>{total_tokens:,}</b> 
    (prompt: {prompt_tokens:,} | completion: {completion_tokens:,})
</div>
"""
