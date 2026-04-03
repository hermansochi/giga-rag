"""
src/document/parsers/csv_parser.py

Парсер CSV-файлов.
Преобразует табличные данные в структурированный текст, пригодный для RAG.
"""

import csv
import io
from typing import List, Tuple

import streamlit as st

from ..base import BaseDocumentParser


class CSVParser(BaseDocumentParser):
    """Парсер для CSV-файлов."""

    def get_supported_extensions(self) -> List[str]:
        """Возвращает список поддерживаемых расширений."""
        return [".csv"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит CSV-файл и преобразует каждую строку в текстовую запись.

        Поддерживает CSV как с заголовками, так и без них.
        """
        pages = []
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            csv_file = io.StringIO(text)
            reader = csv.DictReader(csv_file)

            if not reader.fieldnames:
                # Без заголовков
                csv_file.seek(0)
                reader = csv.reader(csv_file)

                for i, row in enumerate(reader, start=1):
                    if row:
                        row_text = " | ".join(str(cell).strip() for cell in row)
                        pages.append((i, f"Строка {i}: {row_text}"))
            else:
                # С заголовками
                for i, row in enumerate(reader, start=1):
                    if row:
                        row_items = [f"{key}: {value}" for key, value in row.items() if value]
                        row_text = " | ".join(row_items)
                        pages.append((i, f"Строка {i}: {row_text}"))

            if not pages:
                st.error(f"⚠️ CSV файл {filename} пустой или не содержит данных")

            return pages

        except Exception as e:
            st.error(f"❌ Ошибка парсинга CSV {filename}: {e}")
            return []


def parse_csv(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная обёртка для парсинга CSV-файлов.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла.

    Returns:
        List[Tuple[int, str]]: Результат парсинга.
    """
    parser = CSVParser()
    return parser.parse(file_bytes, filename)