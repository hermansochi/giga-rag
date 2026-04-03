"""
src/document/parsers/txt_parser.py

Парсер текстовых файлов (.txt, .text, .md).
Поддерживает несколько кодировок и корректно обрабатывает пустые файлы.
"""

from typing import List, Tuple

import streamlit as st

from ..base import BaseDocumentParser


class TXTParser(BaseDocumentParser):
    """Парсер для текстовых файлов с расширениями .txt, .text, .md."""

    def get_supported_extensions(self) -> List[str]:
        """Возвращает список поддерживаемых расширений."""
        return [".txt", ".text", ".md"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит текстовый файл и возвращает его содержимое как одну страницу.

        Пытается декодировать файл с использованием нескольких кодировок.
        При неудаче использует 'utf-8' с заменой ошибок.
        """
        try:
            text = None
            for encoding in ["utf-8", "utf-8-sig", "cp1251", "koi8-r", "latin1"]:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                text = file_bytes.decode("utf-8", errors="replace")

            text = text.strip()

            if not text:
                st.error(f"⚠️ TXT файл {filename} пустой")
                return []

            return [(1, text)]  # TXT считается одной большой страницей

        except Exception as e:
            st.error(f"❌ Ошибка парсинга TXT {filename}: {e}")
            return []


def parse_txt(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная обёртка для парсинга TXT-файлов.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла.

    Returns:
        List[Tuple[int, str]]: Результат парсинга.
    """
    parser = TXTParser()
    return parser.parse(file_bytes, filename)