"""
src/document/chunker.py

Модуль семантического чанкинга текста.
Использует spaCy (ru_core_news_md) для разбиения по границам предложений.
При недоступности spaCy автоматически переключается на резервный символьный метод.
"""

from typing import List

from src.config import settings

try:
    import spacy

    # Загружаем модель из локальной папки, указанной в настройках
    NLP = spacy.load(settings.SPACY_MODEL_PATH, disable=["ner", "parser", "lemmatizer"])
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
    Основная функция семантического чанкинга текста.

    Старается сохранять целостность предложений.
    При слишком длинных предложениях использует жёсткое разбиение.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if not SPACY_AVAILABLE:
        return _fallback_chunking(text, chunk_size, overlap, min_chunk_size)

    doc = NLP(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sent in doc.sents:
        sentence = sent.text.strip()
        sentence_len = len(sentence)

        if sentence_len > chunk_size * 1.8:
            sub_chunks = _fallback_chunking(sentence, chunk_size, overlap, min_chunk_size)
            chunks.extend(sub_chunks)
            continue

        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)

            # Перекрытие: берём последние ~35% предложений
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


def _fallback_chunking(
    text: str, 
    chunk_size: int, 
    overlap: int, 
    min_chunk_size: int
) -> List[str]:
    """
    Резервный метод чанкинга на основе символов.
    Используется при отсутствии spaCy.
    """
    if not text:
        return []

    chunks = []
    i = 0
    text_len = len(text)

    while i < text_len:
        end = min(i + chunk_size, text_len)
        chunk = text[i:end].strip()

        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        i += chunk_size - overlap

        # Защита от зацикливания при некорректных параметрах
        if overlap >= chunk_size and i < text_len:
            i += max(1, chunk_size // 2)

    return chunks


def smart_chunk(
    text: str, 
    chunk_size: int = 900, 
    overlap: int = 150
) -> List[str]:
    """
    Удобная обёртка для семантического чанкинга.

    Используется в модуле загрузки документов.
    """
    return create_chunks(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=60
    )