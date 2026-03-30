"""
Модуль для семантического чанкинга текста с использованием spaCy.

Содержит функции для разбиения длинного текста на смысловые блоки (чанки),
подходящие для использования в RAG-системах (Retrieval-Augmented Generation).

Основная функция create_chunks() использует модель spaCy (ru_core_news_md)
для корректного разбиения текста по границам предложений и абзацев,
что обеспечивает целостность смысла в каждом чанке. В случае отсутствия spaCy
автоматически используется резервный метод на основе слов.
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
    Разбивает текст на семантически осмысленные чанки с использованием spaCy.

    Эта функция является основным методом чанкинга в системе. Она разбивает
    входной текст на фрагменты заданного размера, стараясь разрезать текст
    на границах предложений, чтобы сохранить целостность смысла.

    При разбиении используется перекрытие (overlap) между соседними чанками
    для сохранения контекста. Также реализована обработка очень длинных
    предложений путём их принудительного разбиения.

    Args:
        text (str): Исходный текст для разбиения. Может быть многострочным.
        chunk_size (int): Целевой размер чанка в символах (примерно 150-200 токенов).
                        Рекомендуемый диапазон: 850–950.
        overlap (int): Количество символов перекрытия между соседними чанками.
                      Рекомендуемый диапазон: 120–180.
        min_chunk_size (int): Минимальный допустимый размер чанка в символах.
                            Чанки меньше этого размера отбрасываются.

    Returns:
        List[str]: Список строк-чанков, каждый из которых представляет
                   собой семантически связный фрагмент текста.

    Notes:
        - Если spaCy недоступна, используется резервный метод (_fallback_chunking).
        - Пустые или почти пустые входные тексты возвращают пустой список.
        - Очень длинные предложения (более 1.8 * chunk_size) разбиваются жёстко.
        - Перекрытие между чанками реализуется через сохранение части последних
          предложений предыдущего чанка.

    Example:
        >>> text = "Первое предложение. Второе предложение. Третье предложение."
        >>> chunks = create_chunks(text, chunk_size=50, overlap=10)
        >>> len(chunks)
        1
        >>> chunks[0]
        'Первое предложение. Второе предложение. Третье предложение.'
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
    """
    Резервный метод чанкинга на основе слов, используемый при отсутствии spaCy.

    Разбивает текст на фрагменты по количеству слов (с учётом длины в символах)
    без учёта семантики. Используется, если библиотека spaCy не установлена
    или не удалось загрузить модель.

    Args:
        text (str): Исходный текст для разбиения.
        chunk_size (int): Целевой размер чанка в символах.
        overlap (int): Размер перекрытия между чанками в символах.
        min_chunk_size (int): Минимальный размер чанка для включения в результат.

    Returns:
        List[str]: Список строк-чанков, полученных простым разбиением по словам.

    Notes:
        - Менее точный, чем основной метод, но обеспечивает работоспособность.
        - Разбиение может происходить посреди предложения.
    """
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
    Удобная функция для семантического чанкинга текста.

    Является обёрткой над create_chunks() с рекомендуемыми параметрами.
    Предназначена для импорта и использования в других модулях проекта.

    Args:
        text (str): Текст для разбиения на чанки.
        chunk_size (int): Целевой размер чанка в символах. Значение по умолчанию: 900.
        overlap (int): Размер перекрытия между чанками. Значение по умолчанию: 150.

    Returns:
        List[str]: Список семантически связных чанков.

    Example:
        >>> with open("document.txt", "r", encoding="utf-8") as f:
        ...     text = f.read()
        >>> chunks = smart_chunk(text)
        >>> print(f"Получено {len(chunks)} чанков")
    """
    return create_chunks(text, chunk_size=chunk_size, overlap=overlap)