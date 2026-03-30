"""
pages/2_Загрузить_файл.py

Загрузка документов в RAG с использованием DTO.
"""

import streamlit as st
import time
import hashlib
import io
from typing import List

from src.config import settings
from src.gigachat import get_gigachat_client
from src.document.parser import parse_document
from src.document.chunker import smart_chunk
from src.models import ParsedDocument, Chunk

# MinIO
from minio import Minio


st.set_page_config(page_title="Загрузка документов", page_icon="📤", layout="wide")
st.title("📤 Загрузка документов в RAG")

st.markdown("Поддерживаемые форматы: **PDF, TXT, CSV, JSON, JSONL**")


# ====================== MinIO клиент ======================
@st.cache_resource
def get_minio_client():
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE
    )


minio_client = get_minio_client()


def ensure_bucket_exists():
    bucket_name = settings.MINIO_BUCKET_NAME
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        st.success(f"✅ Бакет `{bucket_name}` создан")


# ====================== Адаптивная генерация эмбеддингов ======================
def get_embeddings_adaptive(chunks: List[Chunk], client, initial_batch_size: int, filename: str, document_type: str):
    """Генерирует эмбеддинги и сохраняет только новые чанки."""
    if not chunks:
        return 0

    from src.database import get_db_connection, save_chunks
    import hashlib

    batch_size = initial_batch_size
    i = 0
    total = len(chunks)
    saved_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_info = st.empty()

    st.info(f"**Всего чанков: {total}** (файл: {filename})")

    conn = get_db_connection()

    while i < total:
        current_batch = chunks[i:i + batch_size]
        batch_info.info(f"**Текущий пакет: {len(current_batch)} чанков**")

        status_text.text(f"Генерация эмбеддингов: {i}/{total} чанков")

        try:
            texts = [c.text for c in current_batch]
            response = client.embeddings(model=settings.EMBEDDING_MODEL, texts=texts)
            batch_embeddings = [item.embedding for item in response.data]

            new_chunks = []
            new_embeddings = []
            new_metadata = []

            with conn.cursor() as cur:
                for chunk_obj, emb in zip(current_batch, batch_embeddings):
                    chunk_hash = hashlib.sha256(chunk_obj.text.encode('utf-8')).hexdigest()

                    cur.execute("""
                        SELECT id FROM document_chunks 
                        WHERE chunk_hash = %s AND embedding_model = %s
                    """, (chunk_hash, settings.EMBEDDING_MODEL))

                    if not cur.fetchone():
                        new_chunks.append(chunk_obj.text)
                        new_embeddings.append(emb)
                        new_metadata.append({
                            "original_filename": filename,
                            "document_type": document_type,
                            "upload_timestamp": time.time(),
                            "chunk_length": len(chunk_obj.text),
                            "minio_path": filename,
                            **chunk_obj.metadata
                        })

            if new_chunks:
                doc_id = save_chunks(
                    filename=filename,
                    chunks=new_chunks,
                    embeddings=new_embeddings,
                    metadata_list=new_metadata,
                    document_type=document_type
                )
                if doc_id:
                    saved_count += len(new_chunks)

            if batch_size < initial_batch_size and new_chunks:
                old = batch_size
                batch_size = min(batch_size + 2, initial_batch_size)
                if batch_size > old:
                    batch_info.success(f"✅ Пакет увеличен: {old} → {batch_size}")

            i += len(current_batch)
            progress_bar.progress(min(i / total, 1.0))

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "too many requests" in error_str:
                st.warning("429 — Too Many Requests...")
                delay = min(8.0, 2.0 * 2.5) if 'delay' in locals() else 2.0
                old = batch_size
                batch_size = max(1, batch_size // 2)
                batch_info.warning(f"⚠️ Пакет уменьшен: {old} → {batch_size}")
                time.sleep(delay)
            else:
                st.warning(f"Ошибка пакета: {e}")
                batch_size = max(1, batch_size // 2)
                time.sleep(1.5)

    progress_bar.progress(1.0)
    status_text.success(f"✅ Обработка завершена. Новых чанков сохранено: **{saved_count}**")
    batch_info.empty()

    return saved_count


# ====================== Боковая панель ======================
with st.sidebar:
    st.header("Настройки загрузки")

    document_type = st.selectbox(
        "Тип документа",
        options=["manual", "contract", "report", "article", "data", "other"],
        index=0
    )

    chunk_size = st.slider("Размер чанка (символов)", 500, 2000, 950, step=50)
    overlap = st.slider("Перекрытие чанков (символов)", 50, 400, 180, step=10)

    batch_size = st.slider("Начальный размер пакета эмбеддингов", 1, 50, 10, step=1)

    st.divider()
    st.caption(f"Эмбеддинг модель: **{settings.EMBEDDING_MODEL}**")


# ====================== Основная логика ======================
uploaded_files = st.file_uploader(
    "Выберите файлы",
    type=["pdf", "txt", "text", "md", "csv", "json", "jsonl"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Обработать и сохранить в базу", type="primary", use_container_width=True):
        ensure_bucket_exists()
        client = get_gigachat_client()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)
        success_count = 0

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"📄 Обрабатываю: **{uploaded_file.name}** ({idx+1}/{total_files})")

                file_bytes = uploaded_file.read()
                filename = uploaded_file.name

                # 1. Сохраняем оригинал в MinIO
                minio_client.put_object(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=filename,
                    data=io.BytesIO(file_bytes),
                    length=len(file_bytes),
                    content_type=uploaded_file.type or "application/octet-stream"
                )

                # 2. Парсим документ → ParsedDocument
                pages = parse_document(file_bytes, filename)
                if not pages:
                    st.warning(f"Не удалось извлечь текст из {filename}")
                    continue

                # 3. Чанкируем → List[Chunk]
                all_chunks: List[Chunk] = []
                for page_num, page_text in pages:
                    text_chunks = smart_chunk(page_text, chunk_size=chunk_size, overlap=overlap)
                    for j, text in enumerate(text_chunks):
                        all_chunks.append(Chunk(
                            text=text,
                            metadata={
                                "page_number": page_num,
                                "original_filename": filename,
                                "document_type": document_type,
                            },
                            chunk_index=j,
                            document_filename=filename
                        ))

                if not all_chunks:
                    continue

                # 4. Генерация эмбеддингов + сохранение новых чанков
                saved = get_embeddings_adaptive(
                    chunks=all_chunks,
                    client=client,
                    initial_batch_size=batch_size,
                    filename=filename,
                    document_type=document_type
                )

                if saved > 0:
                    success_count += 1
                    st.success(f"✅ Файл `{filename}`: сохранено **{saved}** новых чанков")
                else:
                    st.info(f"ℹ️ Файл `{filename}`: все чанки уже существуют в базе.")

            except Exception as e:
                st.error(f"❌ Ошибка при обработке {uploaded_file.name}: {e}")

            progress_bar.progress((idx + 1) / total_files)

        if success_count > 0:
            st.success(f"🎉 Загрузка завершена! Успешно обработано **{success_count}** из {total_files} файлов.")
            st.balloons()

st.divider()
st.caption("💡 Размер пакета эмбеддингов можно регулировать в боковой панели.")
