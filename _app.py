"""
_app.py

Главная точка входа Streamlit-приложения.
"""

import streamlit as st
from src.config import settings
from src.database import init_vector_db

st.set_page_config(
    page_title="RAG система на GigaChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация базы данных (один раз за сессию)
if "db_fully_initialized" not in st.session_state:
    with st.spinner("Инициализация базы данных... 🛠️"):
        init_vector_db()
    st.session_state.db_fully_initialized = True

st.title("🧠 RAG-система на GigaChat")
st.markdown("### Интеллектуальный помощник с поддержкой документов (PDF, TXT, CSV)")

st.info(f"""
**Текущие настройки:**
- Модель чата: **{settings.GIGACHAT_MODEL}**
- Модель эмбеддингов: **{settings.EMBEDDING_MODEL}**
- Размерность эмбеддингов: **{settings.EMBEDDING_DIM}**
- Поддерживаемые форматы: PDF, TXT, CSV
""")

st.caption("Используйте боковое меню для перехода между страницами.")