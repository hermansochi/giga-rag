# pages/3_Мониторинг.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from src.database import get_db_connection

st.set_page_config(page_title="Мониторинг RAG", page_icon="📊", layout="wide")
st.title("📊 Мониторинг RAG-системы")

# ====================== Загрузка данных ======================
def load_token_usage():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                timestamp,
                total_tokens,
                prompt_tokens,
                completion_tokens
            FROM token_usage_log 
            ORDER BY timestamp DESC
            LIMIT 500
        """, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()


def load_document_stats():
    conn = get_db_connection()
    try:
        # Статистика документов
        stats_df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT document_id) as total_documents,
                COUNT(DISTINCT filename) as unique_files
            FROM document_chunks
        """, conn)

        stats = stats_df.iloc[0] if not stats_df.empty else pd.Series({'total_chunks': 0, 'total_documents': 0})

        # Топ файлов
        top_files = pd.read_sql_query("""
            SELECT 
                filename,
                COUNT(*) as chunks_count,
                MIN(created_at) as first_chunk
            FROM document_chunks 
            GROUP BY filename 
            ORDER BY chunks_count DESC 
            LIMIT 10
        """, conn)

        return stats, top_files
    except Exception:
        return pd.Series({'total_chunks': 0, 'total_documents': 0}), pd.DataFrame()


# ====================== Диагностика ======================
st.subheader("🔍 Диагностика базы данных")

conn = get_db_connection()
try:
    try:
        count_df = pd.read_sql_query("SELECT COUNT(*) as cnt FROM document_chunks", conn)
        if not count_df.empty:
            count = count_df.iloc[0]['cnt']
            st.write(f"Всего записей в таблице `document_chunks`: **{count}**")
        else:
            st.write("Таблица `document_chunks` пустая.")
    except Exception as e:
        st.error(f"Ошибка подсчёта записей: {e}")

    sample = pd.read_sql_query("""
        SELECT filename, chunk_index, page_number, embedding_model 
        FROM document_chunks 
        LIMIT 5
    """, conn)
    if not sample.empty:
        st.write("Пример записей:")
        st.dataframe(sample, use_container_width=True)
    else:
        st.warning("Таблица `document_chunks` пока пустая")
except Exception as e:
    st.error(f"Ошибка диагностики: {e}")

st.divider()

# ====================== Метрики ======================
token_df = load_token_usage()
stats, top_files = load_document_stats()

col1, col2, col3, col4 = st.columns(4)

total_tokens = int(token_df['total_tokens'].sum()) if not token_df.empty else 0
avg_tokens = int(token_df['total_tokens'].mean()) if not token_df.empty and len(token_df) > 0 else 0

total_documents = int(pd.to_numeric(stats.get('total_documents', 0), errors='coerce')) if pd.notna(pd.to_numeric(stats.get('total_documents'), errors='coerce')) else 0
total_chunks = int(pd.to_numeric(stats.get('total_chunks', 0), errors='coerce')) if pd.notna(pd.to_numeric(stats.get('total_chunks'), errors='coerce')) else 0

col1.metric("Всего токенов", f"{total_tokens:,}")
col2.metric("Среднее на запрос", f"{avg_tokens:,}" if avg_tokens > 0 else "0")
col3.metric("Документов в БД", f"{total_documents:,}")
col4.metric("Чанков в БД", f"{total_chunks:,}")

st.divider()

# ====================== Графики ======================
tab1, tab2, tab3 = st.tabs(["📈 Токены и время", "📊 Распределение", "📋 Топ файлов"])

with tab1:
    if not token_df.empty:
        fig1 = px.line(
            token_df.sort_values('timestamp'),
            x='timestamp',
            y=['prompt_tokens', 'completion_tokens', 'total_tokens'],
            title="Расход токенов во времени",
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Пока нет данных по использованию токенов.")

with tab2:
    if not token_df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(token_df, values='total_tokens', names=['prompt_tokens', 'completion_tokens'], hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            fig_hist = px.histogram(token_df, x='total_tokens', nbins=30)
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Нет данных для распределения.")

with tab3:
    st.subheader("Топ-10 файлов по количеству чанков")
    if not top_files.empty and len(top_files) > 0:
        st.dataframe(top_files, use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет загруженных документов.")

# ====================== Сырые данные ======================
with st.expander("📋 Сырые логи токенов (последние 50)"):
    if not token_df.empty:
        display_df = token_df.head(50).copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("Логов пока нет.")

st.caption(f"Последнее обновление: {datetime.now().strftime('%H:%M:%S')}")