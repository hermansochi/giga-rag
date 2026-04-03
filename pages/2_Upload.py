"""
pages/2_Upload_info.py

Загрузка документов в RAG-систему с батчевой генерацией эмбеддингов.
Проверка существования чанков происходит перед генерацией эмбеддингов.
Поддерживаемые форматы получаются динамически из парсеров.
"""

import streamlit as st
import time
import hashlib
import io
from typing import List

from src.config import settings
from src.gigachat import get_gigachat_client
from src.document.parser import parse_document, get_supported_extensions
from src.document.chunker import smart_chunk
from src.models import Chunk

# MinIO
from minio import Minio


st.set_page_config(page_title="Загрузка документов", page_icon="📤", layout="wide")
st.title("📤 Загрузка документов в RAG")

# Получаем поддерживаемые расширения из парсера
supported_extensions = get_supported_extensions()
st.markdown(f"Поддерживаемые форматы: **{', '.join(supported_extensions)}**")


@st.cache_resource
def get_minio_client():
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


minio_client = get_minio_client()


def ensure_bucket_exists():
    """Создаёт бакет, если его нет."""
    bucket_name = settings.MINIO_BUCKET_NAME
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        st.success(f"✅ Бакет `{bucket_name}` создан в MinIO")


def process_file(
    uploaded_file, document_type: str, chunk_size: int, overlap: int, batch_size: int
):
    """Обрабатывает один файл: парсинг → чанкинг → проверка → батчевая генерация эмбеддингов."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()

    # 1. Сохраняем оригинал в MinIO
    minio_client.put_object(
        bucket_name=settings.MINIO_BUCKET_NAME,
        object_name=filename,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type=uploaded_file.type or "application/octet-stream",
    )

    # 2. Парсим документ
    pages = parse_document(file_bytes, filename)
    if not pages:
        st.warning(f"Не удалось извлечь текст из {filename}")
        return 0

    # 3. Создаём чанки (DTO Chunk)
    all_chunks: List[Chunk] = []
    for page_num, page_text in pages:
        texts = smart_chunk(page_text, chunk_size=chunk_size, overlap=overlap)
        for idx, text in enumerate(texts):
            all_chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "page_number": page_num,
                        "original_filename": filename,
                        "document_type": document_type,
                    },
                    chunk_index=idx,
                    document_filename=filename,
                )
            )

    if not all_chunks:
        return 0

    st.info(f"**Файл {filename}:** создано {len(all_chunks)} чанков")

    # 4. Подготовка к обработке
    from src.database import get_db_connection, save_chunks

    client = get_gigachat_client()
    conn = get_db_connection()

    new_texts = []
    new_metadata_list = []
    saved_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_info = st.empty()

    # Обрабатываем чанки пакетами
    for i in range(0, len(all_chunks), batch_size):
        current_batch = all_chunks[i : i + batch_size]
        batch_info.info(
            f"**Пакет {i // batch_size + 1}:** проверяем {len(current_batch)} чанков"
        )

        texts_to_process = []
        metadata_to_process = []

        with conn.cursor() as cur:
            for chunk in current_batch:
                chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

                cur.execute(
                    """
                    SELECT id FROM document_chunks 
                    WHERE chunk_hash = %s AND embedding_model = %s
                """,
                    (chunk_hash, settings.EMBEDDING_MODEL),
                )

                if not cur.fetchone():
                    texts_to_process.append(chunk.text)
                    metadata_to_process.append(
                        {
                            "original_filename": filename,
                            "document_type": document_type,
                            "upload_timestamp": time.time(),
                            "chunk_length": len(chunk.text),
                            "minio_path": filename,
                            **chunk.metadata,
                        }
                    )

        if texts_to_process:
            status_text.text(
                f"Генерация эмбеддингов для пакета ({len(texts_to_process)} новых чанков)..."
            )

            try:
                response = client.embeddings(
                    model=settings.EMBEDDING_MODEL, texts=texts_to_process
                )
                batch_embeddings = [item.embedding for item in response.data]

                doc_id = save_chunks(
                    filename=filename,
                    chunks=texts_to_process,
                    embeddings=batch_embeddings,
                    metadata_list=metadata_to_process,
                    document_type=document_type,
                )

                if doc_id:
                    saved_count += len(texts_to_process)

            except Exception as e:
                st.warning(f"Ошибка генерации эмбеддингов для пакета: {e}")

        progress_bar.progress(min((i + batch_size) / len(all_chunks), 1.0))

    progress_bar.progress(1.0)
    status_text.success(
        f"✅ Файл `{filename}` обработан. Новых чанков сохранено: **{saved_count}**"
    )
    batch_info.empty()

    return saved_count


with st.sidebar:
    st.header("Настройки загрузки")

    document_type = st.selectbox(
        "Тип документа",
        options=["manual", "contract", "report", "article", "data", "other"],
        index=0,
    )

    chunk_size = st.slider("Размер чанка (символов)", 500, 2000, 950, step=50)
    overlap = st.slider("Перекрытие чанков (символов)", 50, 400, 180, step=10)

    batch_size = st.slider(
        "Размер пакета эмбеддингов",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
        help="Сколько чанков проверять и обрабатывать за один запрос к GigaChat",
    )

    st.divider()
    st.caption(f"Эмбеддинг модель: **{settings.EMBEDDING_MODEL}**")


uploaded_files = st.file_uploader(
    "Выберите файлы",
    type=get_supported_extensions(),   # ← теперь динамически
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button(
        "🚀 Обработать и сохранить в базу", type="primary", use_container_width=True
    ):
        ensure_bucket_exists()

        total_files = len(uploaded_files)
        total_new_chunks = 0

        main_progress = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(
                f"📄 Обрабатываю файл: **{uploaded_file.name}** ({idx + 1}/{total_files})"
            )

            new_count = process_file(
                uploaded_file=uploaded_file,
                document_type=document_type,
                chunk_size=chunk_size,
                overlap=overlap,
                batch_size=batch_size,
            )

            total_new_chunks += new_count
            main_progress.progress((idx + 1) / total_files)

        if total_new_chunks > 0:
            st.success(
                f"🎉 Загрузка завершена! Добавлено **{total_new_chunks}** новых чанков."
            )
            st.balloons()
        else:
            st.info("✅ Все чанки из загруженных файлов уже существовали в базе.")

st.divider()
st.caption("💡 Проверка существования чанков происходит перед генерацией эмбеддингов.")