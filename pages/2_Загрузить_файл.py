# pages/2_Загрузить_файл.py
import streamlit as st
import time
import hashlib
from typing import List

from src.config import settings
from src.database import save_chunks
from src.gigachat import get_gigachat_client
from src.document.parser import parse_document, get_supported_extensions
from src.document.chunker import smart_chunk


st.set_page_config(page_title="Загрузка документов", page_icon="📤", layout="wide")
st.title("📤 Загрузка документов в RAG")

st.markdown("Поддерживаемые форматы: **PDF, TXT, CSV**")

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
            st.text_area("", value=short, height=70, disabled=True, key=f"first_{idx}_{hashlib.md5(texts[idx][:50].encode()).hexdigest()[:6]}")

        if total > 3:
            st.markdown("**...**")
            for idx in range(max(2, total-2), total):
                chunk_hash = hashlib.sha256(texts[idx].encode('utf-8')).hexdigest()[:8]
                short = texts[idx][:280] + "..." if len(texts[idx]) > 280 else texts[idx]
                st.markdown(f"**Чанк {idx+1}** (хеш: `{chunk_hash}`)")
                st.text_area("", value=short, height=70, disabled=True, key=f"last_{idx}_{hashlib.md5(texts[idx][:50].encode()).hexdigest()[:6]}")

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
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Обработать и сохранить в базу", type="primary", use_container_width=True):
        client = get_gigachat_client()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)
        success_count = 0
        skipped_count = 0

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"📄 Обрабатываю: **{uploaded_file.name}** ({idx+1}/{total_files})")

                file_bytes = uploaded_file.read()
                pages = parse_document(file_bytes, uploaded_file.name)

                if not pages:
                    st.warning(f"Не удалось извлечь текст из {uploaded_file.name}")
                    continue

                all_chunks = []
                all_metadata = []

                for page_num, page_text in pages:
                    chunks = smart_chunk(page_text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(chunks)
                    for chunk_text in chunks:
                        all_metadata.append({
                            "page_number": page_num,
                            "original_filename": uploaded_file.name,
                            "document_type": document_type,
                            "upload_timestamp": time.time(),
                            "chunk_length": len(chunk_text)
                        })

                if not all_chunks:
                    continue

                status_text.text(f"🔢 Генерирую эмбеддинги для {len(all_chunks)} чанков...")
                embeddings = get_embeddings_adaptive(all_chunks, client, batch_size)

                if not embeddings or len(embeddings) != len(all_chunks):
                    st.error(f"Не удалось получить эмбеддинги для {uploaded_file.name}")
                    continue

                # Сохранение
                doc_id = save_chunks(
                    filename=uploaded_file.name,
                    chunks=all_chunks,
                    embeddings=embeddings,
                    metadata_list=all_metadata,
                    document_type=document_type
                )

                if doc_id:
                    success_count += 1

            except Exception as e:
                st.error(f"❌ Ошибка при обработке {uploaded_file.name}: {e}")

            progress_bar.progress((idx + 1) / total_files)

        # Финальное сообщение
        if success_count > 0:
            st.success(f"🎉 Загрузка завершена! Успешно обработано **{success_count}** из {total_files} файлов.")
            st.balloons()
        else:
            st.info(f"ℹ️ Загрузка не требуется. Все чанки из {total_files} файлов уже существуют в базе.")

st.divider()
st.caption("💡 Размер пакета эмбеддингов можно регулировать в боковой панели.")