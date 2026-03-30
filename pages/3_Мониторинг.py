"""
pages/3_Мониторинг.py

Мониторинг RAG-системы с новым разделом "Таблицы базы данных"
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from collections import defaultdict

from src.database import get_db_connection
from src.config import settings

# MinIO клиент
from minio import Minio


st.set_page_config(
    page_title="Мониторинг",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Мониторинг RAG-системы")

conn = get_db_connection()

# ====================== 1. БАЛАНС И ТОКЕНЫ ======================
st.subheader("💰 Баланс и использование токенов GigaChat")

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                timestamp,
                total_tokens,
                prompt_tokens,
                completion_tokens,
                balance_entries
            FROM token_usage_log 
            ORDER BY timestamp DESC 
            LIMIT 150
        """)
        log_rows = cur.fetchall()

    st.write(f"**Всего записей в логах:** {len(log_rows)}")

    token_data = []
    balance_by_model = defaultdict(list)

    for row in log_rows:
        ts = str(row['timestamp'])[:19] if row['timestamp'] else "Неизвестно"
        total = row['total_tokens'] if row['total_tokens'] is not None else 0

        token_data.append({
            'timestamp': ts,
            'total_tokens': total,
            'prompt_tokens': row['prompt_tokens'] or 0,
            'completion_tokens': row['completion_tokens'] or 0
        })

        if row['balance_entries']:
            try:
                data = row['balance_entries']
                if isinstance(data, str):
                    data = json.loads(data)

                if isinstance(data, dict) and 'balance' in data:
                    for item in data['balance']:
                        model_name = item.get('usage', 'unknown')
                        value = float(item.get('value', 0))
                        balance_by_model[model_name].append({
                            'timestamp': ts,
                            'value': value
                        })
            except:
                pass

    col1, col2, col3 = st.columns(3)
    total_all = sum(item['total_tokens'] for item in token_data)
    avg_tokens = int(sum(item['total_tokens'] for item in token_data) / len(token_data)) if token_data else 0

    col1.metric("Всего токенов использовано", f"{total_all:,}")
    col2.metric("Среднее токенов на запрос", f"{avg_tokens:,}")
    col3.metric("Всего запросов", len(log_rows))

    if token_data:
        df_tokens = pd.DataFrame(token_data)
        fig_tokens = px.line(
            df_tokens,
            x='timestamp',
            y=['prompt_tokens', 'completion_tokens', 'total_tokens'],
            title="Расход токенов во времени",
            template="plotly_white",
            markers=True
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

    st.subheader("📉 Изменение баланса по моделям")

    if balance_by_model:
        for model_name, history in balance_by_model.items():
            if not history:
                continue
            df_model = pd.DataFrame(history)
            fig = px.line(
                df_model,
                x='timestamp',
                y='value',
                title=f"Баланс модели: **{model_name}**",
                template="plotly_white",
                markers=True
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Пока нет данных по балансу моделей.")

except Exception as e:
    st.error(f"Ошибка загрузки логов токенов: {e}")

st.divider()

# ====================== 2. MINIO — ХРАНИЛИЩЕ ======================
st.subheader("📦 MinIO — Хранилище оригинальных файлов")

try:
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE
    )

    bucket_name = settings.MINIO_BUCKET_NAME

    if not minio_client.bucket_exists(bucket_name):
        st.warning(f"⚠️ Бакет `{bucket_name}` не найден.")
        if st.button("Создать бакет documents"):
            try:
                minio_client.make_bucket(bucket_name)
                st.success(f"✅ Бакет `{bucket_name}` создан!")
                st.rerun()
            except Exception as create_e:
                st.error(f"Не удалось создать бакет: {create_e}")
    else:
        objects = list(minio_client.list_objects(bucket_name, recursive=True))

        if objects:
            file_list = []
            total_size = 0

            for obj in objects:
                size_mb = obj.size / (1024 * 1024)
                total_size += obj.size
                file_list.append({
                    "Имя файла": obj.object_name,
                    "Размер (МБ)": round(size_mb, 2),
                    "Дата изменения": obj.last_modified.strftime("%Y-%m-%d %H:%M") if obj.last_modified else "—"
                })

            df_files = pd.DataFrame(file_list)

            col1, col2 = st.columns(2)
            col1.metric("Файлов в хранилище", len(objects))
            col2.metric("Общий объём", f"{total_size / (1024*1024):.2f} МБ")

            st.dataframe(df_files, use_container_width=True, hide_index=True)

            if len(df_files) > 1:
                fig = px.bar(
                    df_files.nlargest(10, "Размер (МБ)"),
                    x="Имя файла",
                    y="Размер (МБ)",
                    title="Топ-10 самых больших файлов"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Бакет существует, но пока пуст.")

except Exception as e:
    st.error(f"Ошибка подключения к MinIO: {e}")

st.divider()

# ====================== 3. ТАБЛИЦЫ БАЗЫ ДАННЫХ ======================
st.subheader("📋 Таблицы базы данных")

try:
    with conn.cursor() as cur:
        # Получаем список всех таблиц
        cur.execute("""
            SELECT 
                t.table_name,
                pg_size_pretty(pg_total_relation_size(t.table_schema || '.' || t.table_name)) AS total_size,
                pg_total_relation_size(t.table_schema || '.' || t.table_name) AS total_size_bytes
            FROM 
                information_schema.tables t
            WHERE 
                t.table_schema = 'public'
            ORDER BY 
                total_size_bytes DESC;
        """)
        tables = cur.fetchall()

        table_stats = []

        for table_row in tables:
            table_name = table_row['table_name']
            
            # Считаем количество записей в таблице
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_result = cur.fetchone()
            row_count = int(count_result['count'])
            table_stats.append({
                "Таблица": table_name,
                "Записей": row_count,
                "Размер таблиц": table_row['total_size']
            })

    if table_stats:
        df_tables = pd.DataFrame(table_stats)
        
        col1, col2 = st.columns(2)
        col1.metric("Всего таблиц", len(table_stats))
        total_rows = sum(row['Записей'] for row in table_stats)
        col2.metric("Всего записей во всех таблицах", f"{total_rows:,}")

        # Красивая таблица
        st.dataframe(
            df_tables.sort_values("Записей", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # График распределения записей по таблицам
        fig = px.bar(
            df_tables.sort_values("Записей", ascending=False),
            x="Таблица",
            y="Записей",
            title="Количество записей по таблицам",
            text="Записей"
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Не удалось получить список таблиц.")

except Exception as e:
    if conn:
        conn.rollback()
    st.error(f"Ошибка при получении информации о таблицах: {e}")

st.divider()

# ====================== 4. ДИАГНОСТИКА ЧАНКОВ ======================
st.subheader("🔍 Диагностика чанков")

try:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = int(cur.fetchone()['count'])

        cur.execute("SELECT COUNT(DISTINCT document_id) FROM document_chunks")
        unique_docs = int(cur.fetchone()['count'])

        cur.execute("SELECT COUNT(DISTINCT filename) FROM document_chunks")
        unique_files = int(cur.fetchone()['count'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего чанков", f"{total_chunks:,}")
    col2.metric("Уникальных документов", f"{unique_docs:,}")
    col3.metric("Уникальных файлов", f"{unique_files:,}")

except Exception as e:
    if conn:
        conn.rollback()
    st.error(f"Ошибка диагностики чанков: {e}")

# ====================== 5. ЛОГ ЧАТА ======================
st.subheader("💬 История чата")

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                user_message,
                LEFT(assistant_response, 100) || '...' as preview,
                model_name,
                use_rag,
                total_tokens,
                response_time_sec,
                timestamp
            FROM chat_logs 
            ORDER BY timestamp DESC 
            LIMIT 50
        """)
        rows = cur.fetchall()

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет записей в логах чата.")

except Exception as e:
    if conn:
        conn.rollback()
    st.error(f"Ошибка загрузки логов чата: {e}")

st.subheader("🔍 Детали RAG-запросов")

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                user_message,
                LEFT(assistant_response, 100) || '...' as answer_preview,
                use_rag,
                retrieved_chunks,
                timestamp
            FROM chat_logs 
            WHERE use_rag = true
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        rows = cur.fetchall()

    if rows:
        for row in rows:
            with st.expander(f"💬 {row['user_message'][:80]}... ({row['timestamp']})"):
                st.write("**Найденные источники:**")
                for i, chunk in enumerate(row['retrieved_chunks']):
                    with st.container():
                        st.caption(f"📄 `{chunk['filename']}` | чанк {chunk['chunk_index']} | dist {chunk['distance']:.4f}")
                        st.text(chunk['text'])
    else:
        st.info("Пока нет RAG-запросов.")
except Exception as e:
    if conn:
        conn.rollback()
    st.error(f"Ошибка загрузки RAG-логов: {e}")

st.divider()

st.caption(f"Последнее обновление: {datetime.now().strftime('%H:%M:%S')}")