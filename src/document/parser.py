"""
src/document/parser.py

Фабрика парсеров документов.
Централизованно управляет выбором и регистрацией парсеров для различных форматов файлов.
"""

from typing import List, Tuple, Dict

import streamlit as st

from .base import BaseDocumentParser
from .parsers.pdf_parser import PDFParser
from .parsers.txt_parser import TXTParser
from .parsers.csv_parser import CSVParser
from .parsers.json_parser import JSONParser


class DocumentParserFactory:
    """
    Фабрика парсеров документов.

    Реализует паттерн "Фабрика" для автоматического выбора подходящего парсера
    по расширению файла.
    """

    _parsers: Dict[str, BaseDocumentParser] = {}

    @classmethod
    def register_parser(cls, parser_class: type) -> None:
        """
        Регистрирует парсер для всех поддерживаемых им расширений.

        Args:
            parser_class (type): Класс парсера, наследующий BaseDocumentParser.
        """
        parser = parser_class()
        for ext in parser.get_supported_extensions():
            cls._parsers[ext] = parser

    @classmethod
    def get_parser(cls, filename: str) -> BaseDocumentParser | None:
        """
        Возвращает подходящий парсер для файла по его расширению.

        Args:
            filename (str): Имя файла.

        Returns:
            BaseDocumentParser | None: Экземпляр парсера или None, если формат не поддерживается.
        """
        ext = BaseDocumentParser.get_file_extension(filename)

        if ext in cls._parsers:
            return cls._parsers[ext]

        # Проверка по окончанию имени файла (для составных расширений, например .jsonl)
        for registered_ext in cls._parsers:
            if filename.lower().endswith(registered_ext):
                return cls._parsers[registered_ext]

        return None

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит файл с помощью подходящего парсера.

        Args:
            file_bytes (bytes): Бинарные данные файла.
            filename (str): Имя файла.

        Returns:
            List[Tuple[int, str]]: Список кортежей (номер страницы/фрагмента, текст).
        """
        parser = cls.get_parser(filename)

        if parser is None:
            st.error(f"❌ Формат файла не поддерживается: {filename}")
            return []

        try:
            return parser.parse(file_bytes, filename)
        except Exception as e:
            st.error(f"❌ Ошибка парсинга файла {filename}: {e}")
            return []


# Регистрация всех парсеров
DocumentParserFactory.register_parser(PDFParser)
DocumentParserFactory.register_parser(TXTParser)
DocumentParserFactory.register_parser(CSVParser)
DocumentParserFactory.register_parser(JSONParser)


def parse_document(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная обёртка для парсинга документа.

    Используется в других модулях проекта.
    """
    return DocumentParserFactory.parse_file(file_bytes, filename)


def get_supported_extensions() -> List[str]:
    """
    Возвращает список всех поддерживаемых расширений файлов.

    Returns:
        List[str]: Отсортированный список расширений.
    """
    exts = set()
    for parser in DocumentParserFactory._parsers.values():
        exts.update(parser.get_supported_extensions())
    return sorted(exts)