# src/document/chunker.py
"""
Семантический чанкер на базе spaCy (ru_core_news_md).
Разбивает текст на чанки с учётом границ предложений и абзацев.
"""

from typing import List
import re

try:
    import spacy
    NLP = spacy.load("ru_core_news_md", disable=["ner", "parser", "lemmatizer"])
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    NLP = None


def create_chunks(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
    min_chunk_size: int = 60
) -> List[str]:
    """
    Разбивает текст на семантически coherent чанки с использованием spaCy.

    Args:
        text: Исходный текст
        chunk_size: Целевой размер чанка в символах (примерно 150-200 токенов)
        overlap: Количество символов перекрытия между чанками
        min_chunk_size: Минимальный размер чанка (меньше — отбрасывается)

    Returns:
        List[str]: Список чанков
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if not SPACY_AVAILABLE:
        return _fallback_chunking(text, chunk_size, overlap, min_chunk_size)

    # Обрабатываем текст через spaCy
    doc = NLP(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sent in doc.sents:
        sentence = sent.text.strip()
        sentence_len = len(sentence)

        # Если одно предложение слишком длинное — режем его жёстко
        if sentence_len > chunk_size * 1.8:
            sub_chunks = _fallback_chunking(sentence, chunk_size, overlap, min_chunk_size)
            chunks.extend(sub_chunks)
            continue

        # Если добавление предложения превышает лимит — сохраняем текущий чанк
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)

            # Создаём перекрытие (берём последние ~30-40% предложений)
            overlap_count = max(1, int(len(current_chunk) * 0.35))
            current_chunk = current_chunk[-overlap_count:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len

    # Добавляем последний чанк
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)

    return chunks


def _fallback_chunking(text: str, chunk_size: int, overlap: int, min_chunk_size: int) -> List[str]:
    """Простой резервный чанкер на основе слов"""
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


# Удобная обёртка
def smart_chunk(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Основная функция для использования извне.
    Рекомендуемые значения: chunk_size=850–950, overlap=120–180
    """
    return create_chunks(text, chunk_size=chunk_size, overlap=overlap)
