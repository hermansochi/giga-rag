"""
src/document/parser.py

Фабрика парсеров документов.
"""

from typing import List, Tuple, Dict
import streamlit as st

from .base import BaseDocumentParser
from .parsers.pdf_parser import PDFParser
from .parsers.txt_parser import TXTParser
from .parsers.csv_parser import CSVParser
from .parsers.json_parser import JSONParser   # ← должен быть этот импорт


class DocumentParserFactory:
    _parsers: Dict[str, BaseDocumentParser] = {}

    @classmethod
    def register_parser(cls, parser_class: type):
        """Регистрирует парсер по его расширениям."""
        parser = parser_class()
        for ext in parser.get_supported_extensions():
            cls._parsers[ext] = parser

    @classmethod
    def get_parser(cls, filename: str) -> BaseDocumentParser | None:
        ext = BaseDocumentParser.get_file_extension(filename)

        if ext in cls._parsers:
            return cls._parsers[ext]

        for registered_ext in cls._parsers:
            if filename.lower().endswith(registered_ext):
                return cls._parsers[registered_ext]

        return None

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
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
DocumentParserFactory.register_parser(CSVParser)
DocumentParserFactory.register_parser(JSONParser)   # ← добавили JSON


def parse_document(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    return DocumentParserFactory.parse_file(file_bytes, filename)


def get_supported_extensions() -> List[str]:
    exts = set()
    for parser in DocumentParserFactory._parsers.values():
        exts.update(parser.get_supported_extensions())
    return sorted(exts)