# src/document/parsers/csv_parser.py
"""
Парсер CSV файлов.
Преобразует табличные данные в удобный текстовый формат для RAG.
"""

import csv
import io
from typing import List, Tuple

from ..base import BaseDocumentParser


class CSVParser(BaseDocumentParser):
    """
    Парсер для CSV-файлов.
    Каждая строка таблицы преобразуется в структурированный текст.
    """

    def get_supported_extensions(self) -> List[str]:
        return [".csv"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит CSV и возвращает список: [(1, "структурированный текст строки"), ...]

        Возвращает каждую строку как отдельную "страницу", чтобы чанкер мог лучше работать.
        """
        pages = []
        try:
            # Декодируем байты в текст
            text = file_bytes.decode("utf-8", errors="replace")

            # Используем StringIO для работы с csv
            csv_file = io.StringIO(text)
            reader = csv.DictReader(csv_file)

            if not reader.fieldnames:
                # Если заголовки не определились — пробуем как обычный csv
                csv_file.seek(0)
                reader = csv.reader(csv_file)

                for i, row in enumerate(reader, start=1):
                    if row:  # пропускаем пустые строки
                        row_text = " | ".join(str(cell).strip() for cell in row)
                        pages.append((i, f"Строка {i}: {row_text}"))
            else:
                # Если есть заголовки — используем DictReader
                for i, row in enumerate(reader, start=1):
                    if row:
                        row_items = [f"{key}: {value}" for key, value in row.items() if value]
                        row_text = " | ".join(row_items)
                        pages.append((i, f"Строка {i}: {row_text}"))

            if not pages:
                print(f"⚠️ CSV файл {filename} пустой или не содержит данных")

            return pages

        except Exception as e:
            print(f"❌ Ошибка парсинга CSV {filename}: {e}")
            return []


# Удобная функция
def parse_csv(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Удобная обёртка для внешнего использования"""
    parser = CSVParser()
    return parser.parse(file_bytes, filename)