"""
Модуль для парсинга CSV-файлов.

Содержит класс CSVParser, реализующий интерфейс BaseDocumentParser,
и вспомогательную функцию parse_csv().

Класс CSVParser преобразует табличные данные из CSV-файлов в структурированный
текстовый формат, пригодный для использования в RAG-системах (Retrieval-Augmented Generation).

Поддерживает CSV-файлы с заголовками и без, автоматически определяя формат.
"""

import csv
import io
from typing import List, Tuple
import streamlit as st

from ..base import BaseDocumentParser


class CSVParser(BaseDocumentParser):
    """
    Парсер для CSV-файлов, преобразующий табличные данные в текст.

    Данный класс отвечает за чтение CSV-файлов и преобразование каждой строки
    таблицы в структурированную текстовую запись. Каждая строка рассматривается
    как отдельная "страница" для последующей обработки чанкером.

    При наличии заголовков в CSV-файле, данные форматируются в виде
    "ключ: значение". В случае отсутствия заголовков, ячейки объединяются
    через разделитель " | ".

    Поддерживаемые расширения файлов:
        - .csv

    Attributes:
        None (нет внутреннего состояния)

    Methods:
        get_supported_extensions() -> List[str]:
            Возвращает список поддерживаемых расширений файлов.

        parse(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
            Основной метод парсинга. Принимает бинарные данные CSV-файла
            и его имя. Возвращает список кортежей, где каждый кортеж содержит
            номер строки (используется как номер страницы) и структурированный
            текст этой строки. Возвращает пустой список в случае ошибки
            или пустого файла.
    """

    def get_supported_extensions(self) -> List[str]:
        """
        Возвращает список расширений файлов, поддерживаемых этим парсером.

        Returns:
            List[str]: Список строк с расширениями, в данном случае ['csv'].
        """
        return [".csv"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит бинарное содержимое CSV-файла и преобразует его в список текстовых записей.

        Метод декодирует байтовые данные в текст с использованием кодировки UTF-8
        и режимом замены ошибок. Затем использует модуль csv для чтения данных.

        Логика обработки:
        1. Если определены заголовки (fieldnames), использует DictReader и форматирует
           строки как "ключ: значение | ключ: значение".
        2. Если заголовки отсутствуют, использует обычный reader и объединяет ячейки
           через " | ".

        Каждая строка таблицы преобразуется в отдельную запись со структурированным текстом,
        что позволяет чанкеру более эффективно обрабатывать табличные данные.

        Args:
            file_bytes (bytes): Бинарные данные CSV-файла для парсинга.
            filename (str): Имя файла (используется для логирования ошибок).

        Returns:
            List[Tuple[int, str]]: Список кортежей, где первый элемент - номер строки (1-n),
            а второй элемент - структурированный текст этой строки. Возвращает пустой список,
            если файл пустой или произошла ошибка.

        Example:
            Для CSV с заголовками:
            Имя,Возраст,Город
            Иван,30,Москва

            Вернёт: [(1, "Строка 1: Имя: Иван | Возраст: 30 | Город: Москва")]

            Для CSV без заголовков:
            Иван,30,Москва

            Вернёт: [(1, "Строка 1: Иван | 30 | Москва")]
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
                st.error(f"⚠️ CSV файл {filename} пустой или не содержит данных")

            return pages

        except Exception as e:
            st.error(f"❌ Ошибка парсинга CSV {filename}: {e}")
            return []


# Удобная функция
def parse_csv(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная обёртка для парсинга CSV-файлов.

    Создаёт экземпляр CSVParser и вызывает его метод parse().

    Эта функция предоставляет простой интерфейс для использования
    парсера без необходимости явного создания объекта.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла для логирования.

    Returns:
        List[Tuple[int, str]]: Результат парсинга — список текстовых записей
        по строкам или пустой список при ошибке.

    Example:
        >>> with open("data.csv", "rb") as f:
        ...     content = f.read()
        >>> pages = parse_csv(content, "data.csv")
        >>> for page_num, text in pages:
        ...     print(f"{page_num}: {text}")
    """
    parser = CSVParser()
    return parser.parse(file_bytes, filename)