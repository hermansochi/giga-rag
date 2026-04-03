"""
pages/4_Управление_RAG.py

Страница управления RAG-системой.
Включает работу с хранилищем MinIO, диагностику чанков и статистику таблиц базы данных.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from minio import Minio
from minio.error import S3Error
import time

from src.config import settings
from src.database import get_db_connection


st.set_page_config(
    page_title="Управление RAG",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Управление RAG-системой")


@st.dialog("⚠️ Подтверждение удаления")
def delete_file_with_confirmation(minio_client, bucket_name: str, object_name: str):
    """Модальное окно подтверждения удаления файла."""
    st.warning(f"""
    Вы действительно хотите **навсегда удалить** файл:

    **{object_name}**

    Из MinIO и **все связанные чанки** из таблицы `document_chunks`?
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Да, удалить навсегда", type="primary"):
            try:
                minio_client.remove_object(bucket_name, object_name)

                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM document_chunks WHERE filename = %s",
                        (object_name,),
                    )
                    conn.commit()

                st.success(
                    f"✅ Файл `{object_name}` и все связанные чанки успешно удалены!"
                )

                st.rerun()

            except S3Error as s3e:
                st.error(f"Ошибка MinIO: {s3e}")
            except Exception as e:
                st.error(f"Ошибка при удалении: {e}")

    with col2:
        if st.button("❌ Отмена"):
            st.rerun()


st.subheader("📦 MinIO — Хранилище оригинальных файлов")

try:
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    bucket_name = settings.MINIO_BUCKET_NAME

    if not minio_client.bucket_exists(bucket_name):
        st.warning(f"⚠️ Бакет `{bucket_name}` не существует.")
        if st.button("Создать бакет `documents`"):
            try:
                minio_client.make_bucket(bucket_name)
                st.success(f"✅ Бакет `{bucket_name}` успешно создан!")
                st.rerun()
            except Exception as create_e:
                st.error(f"Не удалось создать бакет: {create_e}")
    else:
        objects = list(minio_client.list_objects(bucket_name, recursive=True))

        if not objects:
            st.info("Бакет существует, но пока пуст.")
        else:
            file_list = []
            total_size = 0

            for obj in objects:
                size_mb = obj.size / (1024 * 1024)
                total_size += obj.size
                file_list.append(
                    {
                        "object_name": obj.object_name,
                        "size_mb": round(size_mb, 2),
                        "last_modified": obj.last_modified.strftime("%Y-%m-%d %H:%M")
                        if obj.last_modified
                        else "—",
                    }
                )

            df_files = pd.DataFrame(file_list)

            col1, col2 = st.columns(2)
            col1.metric("Всего файлов", len(objects))
            col2.metric("Общий объём", f"{total_size / (1024 * 1024):.2f} МБ")

            for idx, row in df_files.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([4, 2, 1])
                    with col_a:
                        st.write(f"**{row['object_name']}**")
                    with col_b:
                        st.caption(f"{row['size_mb']} МБ | {row['last_modified']}")
                    with col_c:
                        if st.button("🗑️ Удалить", key=f"del_btn_{idx}"):
                            delete_file_with_confirmation(
                                minio_client, bucket_name, row["object_name"]
                            )

            if len(df_files) > 1:
                fig = px.bar(
                    df_files.nlargest(10, "size_mb"),
                    x="object_name",
                    y="size_mb",
                    title="Топ-10 самых больших файлов",
                    labels={"object_name": "Файл", "size_mb": "Размер (МБ)"},
                )
                st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Ошибка подключения к MinIO: {e}")

st.divider()

# Диагностика чанков
st.subheader("🔍 Диагностика чанков")

try:
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = int(cur.fetchone()["count"])

        cur.execute("SELECT COUNT(DISTINCT document_id) FROM document_chunks")
        unique_docs = int(cur.fetchone()["count"])

        cur.execute("SELECT COUNT(DISTINCT filename) FROM document_chunks")
        unique_files = int(cur.fetchone()["count"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего чанков", f"{total_chunks:,}")
    col2.metric("Уникальных документов", f"{unique_docs:,}")
    col3.metric("Уникальных файлов", f"{unique_files:,}")

except Exception as e:
    st.error(f"Ошибка диагностики чанков: {e}")

st.divider()

# Таблицы базы данных
st.subheader("📋 Таблицы базы данных")

try:
    conn = get_db_connection()

    with conn.cursor() as cur:
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
            table_name = table_row["table_name"]
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_result = cur.fetchone()
            row_count = int(count_result["count"])
            table_stats.append(
                {
                    "Таблица": table_name,
                    "Записей": row_count,
                    "Размер таблиц": table_row["total_size"],
                }
            )

    if table_stats:
        df_tables = pd.DataFrame(table_stats)

        col1, col2 = st.columns(2)
        col1.metric("Всего таблиц", len(table_stats))
        total_rows = sum(row["Записей"] for row in table_stats)
        col2.metric("Всего записей во всех таблицах", f"{total_rows:,}")

        st.dataframe(
            df_tables.sort_values("Записей", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            df_tables.sort_values("Записей", ascending=False),
            x="Таблица",
            y="Записей",
            title="Количество записей по таблицам",
            text="Записей",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Не удалось получить список таблиц.")

except Exception as e:
    st.error(f"Ошибка при получении информации о таблицах: {e}")

st.divider()

st.caption("Страница управления хранилищем MinIO и очисткой данных RAG-системы")