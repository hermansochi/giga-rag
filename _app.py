"""
_app.py

Главная точка входа Streamlit-приложения.
Выполняет инициализацию сессии, базы данных и глобальных настроек промптов.
"""

import streamlit as st
import uuid

from src.config import settings
from src.database import init_vector_db


# --- Инициализация сессии ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Инициализация кастомных промптов из настроек
if "custom_base_prompt" not in st.session_state:
    st.session_state.custom_base_prompt = settings.BASE_SYSTEM_PROMT

if "custom_rag_prompt" not in st.session_state:
    st.session_state.custom_rag_prompt = settings.RAG_SYSTEM_PROMT

if "custom_rag_suffix" not in st.session_state:
    st.session_state.custom_rag_suffix = settings.RAG_PROMT_SUFFIX

# Инициализация температур
if "custom_base_temperature" not in st.session_state:
    st.session_state.custom_base_temperature = settings.BASE_TEMPERATURE

if "custom_rag_temperature" not in st.session_state:
    st.session_state.custom_rag_temperature = settings.RAG_TEMPERATURE


# --- Инициализация базы данных ---
if "db_fully_initialized" not in st.session_state:
    with st.spinner("Инициализация базы данных... 🛠️"):
        init_vector_db()
    st.session_state.db_fully_initialized = True


st.title("🧠 RAG-система на GigaChat")
st.markdown("### Интеллектуальный помощник с поддержкой документов")


# --- Редактируемые настройки промптов ---
with st.expander("⚙️ Настройки промптов и температур", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Системный промпт (без RAG)**")
        base_prompt = st.text_area(
            "Промпт для режима без поиска по документам:",
            value=st.session_state.custom_base_prompt,
            height=150,
            key="edit_base_prompt",
        )

    with col2:
        st.markdown("**Системный промпт (с RAG)**")
        rag_prompt = st.text_area(
            "Промпт для режима с поиском по документам:",
            value=st.session_state.custom_rag_prompt,
            height=150,
            key="edit_rag_prompt",
        )

    rag_suffix = st.text_input(
        "Дополнительный суффикс к вопросу в RAG-режиме",
        value=st.session_state.custom_rag_suffix,
        key="edit_rag_suffix",
    )

    col3, col4 = st.columns(2)
    with col3:
        base_temp = st.slider(
            "Температура (без RAG)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.custom_base_temperature,
            step=0.05,
            key="edit_base_temp",
        )
    with col4:
        rag_temp = st.slider(
            "Температура (с RAG)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.custom_rag_temperature,
            step=0.05,
            key="edit_rag_temp",
        )

    if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
        st.session_state.custom_base_prompt = base_prompt.strip()
        st.session_state.custom_rag_prompt = rag_prompt.strip()
        st.session_state.custom_rag_suffix = rag_suffix.strip()
        st.session_state.custom_base_temperature = base_temp
        st.session_state.custom_rag_temperature = rag_temp

        st.success("✅ Настройки обновлены (действуют до перезагрузки приложения)")

        st.info("""
        🔧 Чтобы сохранить настройки **навсегда**, добавьте следующие строки в файл `.env`:
        """)
        st.code(
            f"""
BASE_SYSTEM_PROMT={base_prompt.strip()}
RAG_SYSTEM_PROMT={rag_prompt.strip()}
RAG_PROMT_SUFFIX={rag_suffix.strip()}
BASE_TEMPERATURE={base_temp}
RAG_TEMPERATURE={rag_temp}
            """.strip(),
            language="env",
        )


# --- Отображение текущих настроек ---
st.info(f"""
**Текущие настройки:**
- Модель чата: **{settings.GIGACHAT_MODEL}**
- Модель эмбеддингов: **{settings.EMBEDDING_MODEL}**
- Размерность эмбеддингов: **{settings.EMBEDDING_DIM}**
- Системный промпт (без RAG): **{st.session_state.custom_base_prompt[:80]}...**
- Системный промпт (с RAG): **{st.session_state.custom_rag_prompt[:80]}...**
- Суффикс RAG: **{st.session_state.custom_rag_suffix}**
- Температура (без RAG): **{st.session_state.custom_base_temperature}**
- Температура (с RAG): **{st.session_state.custom_rag_temperature}**
- Поддерживаемые форматы: PDF, TXT, CSV, JSON, JSONL
""")


st.caption("Используйте боковое меню для перехода между страницами.")