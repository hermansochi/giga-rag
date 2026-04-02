"""
pages/1_Чат.py — Чат с чистыми метриками
"""

import streamlit as st
import time

from src.gigachat import (
    generate_with_gigachat,
    get_available_models,
    get_reranker_options
)
from src.config import settings
from src.database import log_chat_interaction

st.set_page_config(page_title="Чат", page_icon="🧠", layout="wide")
st.title("💬 Чат")

with st.sidebar:
    st.header("⚙️ Настройки чата")
    
    use_rag: bool = st.checkbox("Использовать RAG", value=True)
    
    available_models: list[str] = get_available_models() or [settings.GIGACHAT_MODEL]
    
    default_index = 0
    try:
        if "GigaChat-2-Max" in available_models:
            default_index = available_models.index("GigaChat-2-Max")
    except Exception:
        default_index = 0

    model_name: str = st.selectbox(
        "Модель GigaChat", 
        options=available_models, 
        index=default_index
    )
    
    # Выбор реранкинга показываем только если включён RAG
    reranker_options = get_reranker_options()
    
    if use_rag:
        selected_reranker: str = st.selectbox(
            "Тип реранкинга",
            options=list(reranker_options.keys()),
            format_func=lambda x: reranker_options[x],
            index=0
        )
    else:
        selected_reranker: str = "none"  # принудительно без реранкинга
        st.caption("🔹 Реранкинг отключён (режим без RAG)")

    st.caption(f"Эмбеддинг модель: **{settings.EMBEDDING_MODEL}**")


# История чата
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


if prompt := st.chat_input("Задайте вопрос..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        
        spinner_text = "Анализирую документы..." if use_rag else "Думаю над ответом..."
        with st.spinner(spinner_text):
            response_html, metrics = generate_with_gigachat(
                prompt=prompt,
                model_name=model_name,
                use_rag=use_rag,
                reranker_type=selected_reranker if use_rag else None
            )
            st.markdown(response_html, unsafe_allow_html=True)

        response_time = time.time() - start_time

    st.session_state.messages.append({"role": "assistant", "content": response_html})

    # ====================== ЛОГИРОВАНИЕ ======================
    try:
        log_chat_interaction(
            user_message=prompt,
            assistant_response=response_html,
            model_name=metrics["model_name"],
            use_rag=metrics["use_rag"],
            reranker_type=metrics.get("reranker_type"),
            total_tokens=metrics["total_tokens"],
            prompt_tokens=metrics["prompt_tokens"],
            completion_tokens=metrics["completion_tokens"],
            response_time=response_time,
            metadata={
                "session_id": st.session_state.get("session_id", "unknown"),
                "page": "chat"
            }
        )

        st.caption(
            f"⏱ Время: **{response_time:.2f} сек** | "
            f"📊 Токены: **{metrics['total_tokens']:,}** "
            f"(prompt: {metrics['prompt_tokens']:,} | completion: {metrics['completion_tokens']:,})"
        )
    except Exception as e:
        st.warning(f"⚠️ Не удалось сохранить лог: {e}")
