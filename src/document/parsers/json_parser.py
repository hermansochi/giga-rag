"""
src/document/parsers/json_parser.py

Парсер для JSON и JSON Lines (.jsonl) файлов.
Поддерживает датасеты в обоих форматах.
"""

import json
from typing import List, Tuple
import streamlit as st

from ..base import BaseDocumentParser


class JSONParser(BaseDocumentParser):
    """
    Парсер JSON и JSONL файлов.
    Поддерживает:
    - .json: массив объектов или объект с массивом
    - .jsonl: каждая строка — отдельный JSON объект
    """

    def get_supported_extensions(self) -> List[str]:
        return [".json", ".jsonl"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит JSON / JSONL файл и возвращает список (page_number, text)
        """
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            pages: List[Tuple[int, str]] = []

            # Определяем формат по расширению
            is_jsonl = filename.lower().endswith('.jsonl')

            if is_jsonl:
                # JSON Lines формат — каждая строка = отдельный JSON
                lines = text.strip().splitlines()
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            item_text = json.dumps(item, ensure_ascii=False, indent=2)
                            pages.append((i, f"Запись {i}:\n{item_text}"))
                        else:
                            pages.append((i, f"Запись {i}: {str(item)}"))
                    except json.JSONDecodeError:
                        st.warning(f"Строка {i} не является валидным JSON")
                        pages.append((i, f"Запись {i} (невалидный JSON): {line[:200]}..."))

            else:
                # Обычный .json файл
                data = json.loads(text)

                if isinstance(data, list):
                    for i, item in enumerate(data, 1):
                        if isinstance(item, dict):
                            item_text = json.dumps(item, ensure_ascii=False, indent=2)
                            pages.append((i, f"Запись {i}:\n{item_text}"))
                        else:
                            pages.append((i, f"Запись {i}: {str(item)}"))

                elif isinstance(data, dict):
                    found = False
                    common_keys = ['data', 'items', 'records', 'rows', 'results', 'entries', 'values', 'list']

                    for key in common_keys:
                        if key in data and isinstance(data[key], list):
                            for i, item in enumerate(data[key], 1):
                                if isinstance(item, dict):
                                    item_text = json.dumps(item, ensure_ascii=False, indent=2)
                                    pages.append((i, f"Запись {i} ({key}):\n{item_text}"))
                                else:
                                    pages.append((i, f"Запись {i}: {str(item)}"))
                            found = True
                            break

                    if not found:
                        full_text = json.dumps(data, ensure_ascii=False, indent=2)
                        pages.append((1, f"JSON объект:\n{full_text}"))

                else:
                    pages.append((1, str(data)))

            if not pages:
                st.warning(f"Файл {filename} не содержит данных.")
                return []

            return pages

        except json.JSONDecodeError as e:
            st.error(f"❌ Некорректный JSON формат в файле {filename}: {e}")
            return []
        except Exception as e:
            st.error(f"❌ Ошибка парсинга файла {filename}: {e}")
            return []


# Удобная обёртка
def parse_json(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    parser = JSONParser()
    return parser.parse(file_bytes, filename)