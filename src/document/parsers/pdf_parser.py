"""
src/document/parsers/pdf_parser.py

Парсер PDF-файлов с использованием PyMuPDF (fitz).
Извлекает текст со всех страниц и сохраняет нумерацию.
"""

import streamlit as st
from typing import List, Tuple
import fitz  # PyMuPDF

from ..base import BaseDocumentParser


class PDFParser(BaseDocumentParser):
    """Парсер для PDF-документов."""

    def get_supported_extensions(self) -> List[str]:
        """Возвращает список поддерживаемых расширений."""
        return [".pdf"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит PDF-файл и возвращает текст по страницам.

        Пропускает пустые страницы (менее 10 символов).
        """
        pages = []

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()

                if text and len(text) > 10:
                    pages.append((page_num + 1, text))

            doc.close()

            if not pages:
                st.error(f"⚠️ В PDF {filename} не удалось извлечь текст")

            return pages

        except Exception as e:
            st.error(f"❌ Ошибка парсинга PDF {filename}: {e}")
            return []


def parse_pdf(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная обёртка для парсинга PDF-файлов.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла.

    Returns:
        List[Tuple[int, str]]: Список кортежей (номер страницы, текст страницы).
    """
    parser = PDFParser()
    return parser.parse(file_bytes, filename)