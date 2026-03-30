"""
pages/1_Чат.py

Главный чат-интерфейс RAG-системы на GigaChat.

Изменения в этой версии:
- Убраны смайлики из спиннеров (теперь чистый текст)
- Заголовок страницы: просто "💬 Чат"
- Клиент GigaChat создаётся тихо (без сообщений)
- Реальные метрики токенов в обоих режимах
"""

import streamlit as st
from typing import Optional
import re
import time

from src.gigachat import (
    generate_with_gigachat,
    get_available_models,
    get_reranker_options
)
from src.config import settings
from src.database import log_chat_interaction

st.set_page_config(
    page_title="Чат",
    page_icon="🧠",
    layout="wide"
)

st.title("💬 Чат")


with st.sidebar:
    st.header("⚙️ Настройки чата")
    
    use_rag: bool = st.checkbox(
        "Использовать RAG",
        value=False,
        help="Если выключено — модель отвечает только своими знаниями. "
             "Если включено — ищет информацию в загруженных документах."
    )
    
    available_models: list[str] = get_available_models() or [settings.GIGACHAT_MODEL]
    
    # Умный выбор модели по умолчанию — GigaChat-2-Max
    default_index = 0
    try:
        if "GigaChat-2-Max" in available_models:
            default_index = available_models.index("GigaChat-2-Max")
        elif any("GigaChat-2-Max" in model for model in available_models):
            for i, model in enumerate(available_models):
                if "GigaChat-2-Max" in model:
                    default_index = i
                    break
    except Exception:
        default_index = 0

    model_name: str = st.selectbox(
        "Модель GigaChat",
        options=available_models,
        index=default_index,
        help="По умолчанию выбрана самая мощная модель — GigaChat-2-Max"
    )
    
    reranker_options = get_reranker_options()
    
    selected_reranker: str = st.selectbox(
        "Тип реранкинга",
        options=list(reranker_options.keys()),
        format_func=lambda x: reranker_options[x],
        index=0,
        help="Пока лучше оставлять 'Без реранкера'"
    )

    st.caption(f"Эмбеддинг модель: **{settings.EMBEDDING_MODEL}**")


# История чата
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)


# Обработка нового вопроса
if prompt := st.chat_input("Задайте вопрос..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        with st.spinner(
            "Анализирую документы..." if use_rag else "Думаю над ответом..."
        ):
            response = generate_with_gigachat(
                prompt=prompt,
                model_name=model_name,
                use_rag=use_rag,
                reranker_type=selected_reranker if use_rag else None
            )
            
            st.markdown(response, unsafe_allow_html=True)

        response_time = time.time() - start_time

    st.session_state.messages.append({"role": "assistant", "content": response})

    # 🔽 Добавляем логирование
    try:

        # Извлекаем метрики из ответа (они в HTML-блоке)

        tokens_match = re.search(r"Токены: <b>(\d{1,3}(?:,\d{3})*)</b>", response)
        total_tokens = int(tokens_match.group(1).replace(",", "")) if tokens_match else 0

        prompt_match = re.search(r"prompt: (\d{1,3}(?:,\d{3})*)", response)
        prompt_tokens = int(prompt_match.group(1).replace(",", "")) if prompt_match else 0

        completion_match = re.search(r"completion: (\d{1,3}(?:,\d{3})*)", response)
        completion_tokens = int(completion_match.group(1).replace(",", "")) if completion_match else 0

        log_chat_interaction(
            user_message=prompt,
            assistant_response=response,
            model_name=model_name,
            use_rag=use_rag,
            reranker_type=selected_reranker if use_rag else None,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_time=response_time,
            metadata={
                "session_id": st.session_state.get("session_id", "unknown"),
                "page": "chat"
            }
        )
    except Exception as e:
        st.warning(f"⚠️ Логирование чата не удалось: {e}")