"""
Модуль для парсинга JSON и JSON Lines (.jsonl) файлов.

Содержит класс JSONParser, реализующий интерфейс BaseDocumentParser,
и вспомогательную функцию parse_json().

Класс JSONParser поддерживает два формата:
- Обычные JSON-файлы (.json), содержащие массив объектов или объект с вложенным массивом.
- JSON Lines (.jsonl), где каждая строка представляет собой отдельный JSON-объект.

Цель парсера — преобразовать структурированные JSON-данные в текстовый формат,
пригодный для RAG-систем (Retrieval-Augmented Generation), где каждая запись
(объект) рассматривается как отдельная "страница".
"""

import json
from typing import List, Tuple
import streamlit as st

from ..base import BaseDocumentParser


class JSONParser(BaseDocumentParser):
    """
    Парсер для JSON и JSON Lines (.jsonl) файлов.

    Этот класс отвечает за обработку двух форматов JSON-данных:
    1. Обычный JSON (.json) — файл может содержать массив объектов или
       объект, в котором один из ключей содержит массив данных.
    2. JSON Lines (.jsonl) — каждая строка файла является отдельным
       валидным JSON-объектом.

    Парсер преобразует каждую запись (объект) в структурированный текст,
    который возвращается как отдельная "страница". Это позволяет чанкеру
    эффективно обрабатывать каждую запись независимо.

    Поддерживаемые расширения файлов:
        - .json
        - .jsonl

    Attributes:
        None (нет внутреннего состояния)

    Methods:
        get_supported_extensions() -> List[str]:
            Возвращает список поддерживаемых расширений файлов.

        parse(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
            Основной метод парсинга. Принимает бинарные данные JSON-файла
            и его имя. Возвращает список кортежей, где каждый кортеж содержит
            номер записи (1-n) и её текстовое представление. Возвращает
            пустой список в случае ошибки или отсутствия данных.
    """

    def get_supported_extensions(self) -> List[str]:
        """
        Возвращает список расширений файлов, поддерживаемых этим парсером.

        Returns:
            List[str]: Список строк с расширениями ['.json', '.jsonl'].
        """
        return [".json", ".jsonl"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит бинарное содержимое JSON или JSONL файла и преобразует его в список текстовых записей.

        Логика парсинга зависит от расширения файла:

        1. Для .jsonl файлов:
           - Файл разбивается на строки.
           - Каждая непустая ��трока парсится как отдельный JSON-объект.
           - Объекты преобразуются в текст (с отступами для словарей).

        2. Для .json файлов:
           - Если корневой элемент — массив, каждая его запись обрабатывается как отдельная страница.
           - Если корневой элемент — словарь, парсер ищет в нём вложенные массивы
             по распространённым ключам (data, items, records и т.д.) и обрабатывает их элементы.
           - Если массив не найден, весь объект помещается на одну страницу.

        Каждая валидная запись преобразуется в структурированный текст с префиксом "Запись N".
        Невалидные строки в .jsonl помечаются как "невалидный JSON".

        Args:
            file_bytes (bytes): Бинарные данные JSON-файла для парсинга.
            filename (str): Имя файла (используется для логирования ошибок и определения формата).

        Returns:
            List[Tuple[int, str]]: Список кортежей (номер_записи, текст_записи).
            Возвращает пустой список, если файл пустой, содержит ошибки или данные не найдены.

        Example:
            Для .jsonl:
            {"name": "Иван", "age": 30}
            {"name": "Мария", "age": 25}

            Вернёт: [
                (1, "Запись 1:\n{\n  \"name\": \"Иван\",\n  \"age\": 30\n}"),
                (2, "Запись 2:\n{\n  \"name\": \"Мария\",\n  \"age\": 25\n}")
            ]

            Для .json массива:
            [{"name": "Иван"}, {"name": "Мария"}]

            Результат аналогичен.
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
    """
    Удобная обёртка для парсинга JSON-файлов.

    Создаёт экземпляр JSONParser и вызывает его метод parse().

    Эта функция предоставляет простой интерфейс для использования
    парсера без необходимости явного создания объекта.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла для логирования.

    Returns:
        List[Tuple[int, str]]: Результат парсинга — список текстовых записей
        по объектам или пустой список при ошибке.

    Example:
        >>> with open("data.json", "rb") as f:
        ...     content = f.read()
        >>> pages = parse_json(content, "data.json")
        >>> for page_num, text in pages:
        ...     print(f"{page_num}: {text}")
    """
    parser = JSONParser()
    return parser.parse(file_bytes, filename)