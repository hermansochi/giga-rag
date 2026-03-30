"""
pages/2_Загрузить_файл.py

Загрузка документов в RAG с сохранением оригиналов в MinIO (S3).
Если файл PDF — сохраняем и оригинальный PDF, и извлечённый текст.
"""

import streamlit as st
import time
import hashlib
import io
from typing import List

from src.config import settings
from src.database import save_chunks
from src.gigachat import get_gigachat_client
from src.document.parser import parse_document
from src.document.chunker import smart_chunk

# MinIO
from minio import Minio


st.set_page_config(page_title="Загрузка документов", page_icon="📤", layout="wide")
st.title("📤 Загрузка документов в RAG")

st.markdown("Поддерживаемые форматы: **PDF, TXT, CSV**")


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
    """Создаёт бакет, если его нет."""
    bucket_name = settings.MINIO_BUCKET_NAME
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        st.success(f"✅ Бакет `{bucket_name}` создан в MinIO")


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

    batch_size = st.slider(
        "Начальный размер пакета эмбеддингов",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Сколько чанков отправлять в одном запросе к GigaChat"
    )

    st.divider()
    st.caption(f"Эмбеддинг модель: **{settings.EMBEDDING_MODEL}**")


# ====================== Адаптивная генерация эмбеддингов ======================
def get_embeddings_adaptive(texts: List[str], client, initial_batch_size: int):
    if not texts:
        return []

    batch_size = initial_batch_size
    embeddings = []
    i = 0
    delay = 0.5
    total = len(texts)

    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_info = st.empty()

    st.info(f"**Всего сгенерировано чанков: {total}**")

    with st.expander(f"📋 Превью чанков (первые и последние)", expanded=False):
        for idx in range(min(2, total)):
            chunk_hash = hashlib.sha256(texts[idx].encode('utf-8')).hexdigest()[:8]
            short = texts[idx][:280] + "..." if len(texts[idx]) > 280 else texts[idx]
            st.markdown(f"**Чанк {idx+1}** (хеш: `{chunk_hash}`)")
            st.text_area("", value=short, height=70, disabled=True, key=f"first_{idx}")

        if total > 3:
            st.markdown("**...**")
            for idx in range(max(2, total-2), total):
                chunk_hash = hashlib.sha256(texts[idx].encode('utf-8')).hexdigest()[:8]
                short = texts[idx][:280] + "..." if len(texts[idx]) > 280 else texts[idx]
                st.markdown(f"**Чанк {idx+1}** (хеш: `{chunk_hash}`)")
                st.text_area("", value=short, height=70, disabled=True, key=f"last_{idx}")

    while i < total:
        current_batch = texts[i:i + batch_size]
        batch_info.info(f"**Текущий размер пакета: {len(current_batch)}** чанков")

        status_text.text(f"Генерация эмбеддингов: {i}/{total} чанков")

        try:
            response = client.embeddings(model=settings.EMBEDDING_MODEL, texts=current_batch)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            if batch_size < initial_batch_size:
                old = batch_size
                batch_size = min(batch_size + 2, initial_batch_size)
                if batch_size > old:
                    batch_info.success(f"✅ Пакет увеличен: {old} → {batch_size}")

            i += len(current_batch)
            progress_bar.progress(min(i / total, 1.0))
            delay = max(0.3, delay * 0.8)

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "too many requests" in error_str:
                st.warning("429 — Too Many Requests. Увеличиваем задержку...")
                delay = min(delay * 2.5, 8.0)
                old = batch_size
                batch_size = max(1, batch_size // 2)
                batch_info.warning(f"⚠️ Пакет уменьшен: {old} → {batch_size}")
                time.sleep(delay)
            else:
                st.warning(f"Ошибка при пакете {len(current_batch)} чанков: {e}")
                old = batch_size
                batch_size = max(1, batch_size // 2)
                batch_info.warning(f"⚠️ Пакет уменьшен: {old} → {batch_size}")
                time.sleep(1.5)

    progress_bar.progress(1.0)
    status_text.success(f"✅ Эмбеддинги успешно получены для всех {total} чанков")
    batch_info.empty()
    return embeddings


# ====================== Основная логика ======================
uploaded_files = st.file_uploader(
    "Выберите файлы",
    type=["pdf", "txt", "text", "md", "csv", "json","jsonl"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Обработать и сохранить в базу", type="primary", use_container_width=True):
        ensure_bucket_exists()   # создаём бакет, если нужно
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

                # 1. Сохраняем оригинальный файл в MinIO
                minio_client.put_object(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=filename,
                    data=io.BytesIO(file_bytes),
                    length=len(file_bytes),
                    content_type=uploaded_file.type or "application/octet-stream"
                )

                # 2. Парсим документ
                pages = parse_document(file_bytes, filename)

                if not pages:
                    st.warning(f"Не удалось извлечь текст из {filename}")
                    continue

                all_chunks = []
                all_metadata = []

                for page_num, page_text in pages:
                    chunks = smart_chunk(page_text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(chunks)
                    for chunk_text in chunks:
                        all_metadata.append({
                            "page_number": page_num,
                            "original_filename": filename,
                            "document_type": document_type,
                            "upload_timestamp": time.time(),
                            "chunk_length": len(chunk_text),
                            "minio_path": filename   # ссылка на оригинал в MinIO
                        })

                if not all_chunks:
                    continue

                status_text.text(f"🔢 Генерирую эмбеддинги для {len(all_chunks)} чанков...")
                embeddings = get_embeddings_adaptive(all_chunks, client, batch_size)

                if not embeddings or len(embeddings) != len(all_chunks):
                    st.error(f"Не удалось получить эмбеддинги для {filename}")
                    continue

                # 3. Сохранение чанков в БД
                doc_id = save_chunks(
                    filename=filename,
                    chunks=all_chunks,
                    embeddings=embeddings,
                    metadata_list=all_metadata,
                    document_type=document_type
                )

                if doc_id:
                    success_count += 1
                    st.success(f"✅ Файл `{filename}` успешно сохранён в MinIO и базу")

            except Exception as e:
                st.error(f"❌ Ошибка при обработке {uploaded_file.name}: {e}")

            progress_bar.progress((idx + 1) / total_files)

        if success_count > 0:
            st.success(f"🎉 Загрузка завершена! Успешно обработано **{success_count}** из {total_files} файлов.")
            st.balloons()
        else:
            st.info("Все чанки уже существовали в базе.")

st.divider()
st.caption("💡 Размер пакета эмбеддингов можно регулировать в боковой панели.")