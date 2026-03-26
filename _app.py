# _app.py
import streamlit as st
from src.config import settings
from src.database import init_vector_db

st.set_page_config(
    page_title="RAG система на GigaChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация базы данных (защищена от повторных вызовов)
if "db_fully_initialized" not in st.session_state:
    with st.spinner("Инициализация базы данных..."):
        init_vector_db()

st.title("🧠 RAG-система на GigaChat")
st.markdown("### Интеллектуальный помощник с поддержкой документов (PDF, TXT, CSV)")



st.info(f"""
**Текущие настройки:**
- Модель чата: **{settings.GIGACHAT_MODEL}**
- Модель эмбеддингов: **{settings.EMBEDDING_MODEL}**
- Размерность эмбеддингов: **{settings.EMBEDDING_DIM}**
- Реранкинг по умолчанию: **{settings.RERANKER_TYPE}**
- Поддерживаемые форматы: PDF, TXT, CSV
""")

st.caption("Используйте боковое меню для перехода между страницами.")