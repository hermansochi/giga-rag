# src/document/parser.py
"""
Фабрика парсеров документов.
Автоматически выбирает нужный парсер по расширению файла.
"""

from typing import List, Tuple, Dict
import streamlit as st

from .base import BaseDocumentParser
from .parsers.pdf_parser import PDFParser
from .parsers.txt_parser import TXTParser
from .parsers.csv_parser import CSVParser   # ← добавили


class DocumentParserFactory:
    """
    Фабрика парсеров.
    """

    _parsers: Dict[str, BaseDocumentParser] = {}

    @classmethod
    def register_parser(cls, parser_class: type):
        """Регистрирует парсер по его поддерживаемым расширениям"""
        parser = parser_class()
        for ext in parser.get_supported_extensions():
            cls._parsers[ext] = parser

    @classmethod
    def get_parser(cls, filename: str) -> BaseDocumentParser | None:
        """Возвращает подходящий парсер по имени файла"""
        ext = BaseDocumentParser.get_file_extension(filename)

        if ext in cls._parsers:
            return cls._parsers[ext]

        # Дополнительная проверка по концу имени файла
        for registered_ext in cls._parsers:
            if filename.lower().endswith(registered_ext):
                return cls._parsers[registered_ext]

        return None

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """Главный метод парсинга любого поддерживаемого документа"""
        parser = cls.get_parser(filename)

        if parser is None:
            st.error(f"❌ Формат файла не поддерживается: {filename}")
            return []

        try:
            return parser.parse(file_bytes, filename)
        except Exception as e:
            st.error(f"❌ Ошибка парсинга файла {filename}: {e}")
            return []


# ====================== Регистрация всех парсеров ======================

DocumentParserFactory.register_parser(PDFParser)
DocumentParserFactory.register_parser(TXTParser)
DocumentParserFactory.register_parser(CSVParser)   # ← добавили CSV


# Удобные функции для использования в других модулях
def parse_document(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Универсальная функция парсинга"""
    return DocumentParserFactory.parse_file(file_bytes, filename)


def get_supported_extensions() -> List[str]:
    """Возвращает список всех поддерживаемых расширений"""
    exts = set()
    for parser in DocumentParserFactory._parsers.values():
        exts.update(parser.get_supported_extensions())
    return sorted(exts)