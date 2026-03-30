"""
Модуль, реализующий фабрику парсеров документов.

Содержит класс DocumentParserFactory, который отвечает за выбор
и управление соответствующим парсером в зависимости от расширения файла.

Позволяет централизованно обрабатывать различные форматы документов
(PDF, TXT, CSV, JSON и др.) через единый интерфейс, обеспечивая гибкость
и расширяемость системы парсинга документов в RAG-приложении.
"""

from typing import List, Tuple, Dict
import streamlit as st

from .base import BaseDocumentParser
from .parsers.pdf_parser import PDFParser
from .parsers.txt_parser import TXTParser
from .parsers.csv_parser import CSVParser
from .parsers.json_parser import JSONParser   # ← должен быть этот импорт


class DocumentParserFactory:
    """
    Фабрика парсеров документов.

    Централизованно управляет экземплярами парсеров и выбирает подходящий
    парсер для файла на основе его расширения.

    Реализует паттерн "Фабрика", позволяя добавлять новые форматы
    документов без изменения основной логики приложения.

    Attributes:
        _parsers (Dict[str, BaseDocumentParser]): Словарь, сопоставляющий
            расширения файлов с экземплярами парсеров. Заполняется при регистрации.

    Methods:
        register_parser(parser_class: type) -> None:
            Регистрирует новый парсер в фабрике.

        get_parser(filename: str) -> BaseDocumentParser | None:
            Возвращает подходящий парсер для указанного файла.

        parse_file(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
            Парсит файл с использованием подходящего парсера.
    """

    _parsers: Dict[str, BaseDocumentParser] = {}

    @classmethod
    def register_parser(cls, parser_class: type):
        """
        Регистрирует парсер в фабрике по его поддерживаемым расширениям.

        Создаёт экземпляр указанного класса парсера и регистрирует его
        для каждого расширения, возвращаемого методом get_supported_extensions().

        Если для одного расширения зарегистрировано несколько парсеров,
        будет использоваться последний зарегистрированный.

        Args:
            parser_class (type): Класс парсера, унаследованный от BaseDocumentParser.

        Example:
            >>> DocumentParserFactory.register_parser(PDFParser)
            >>> DocumentParserFactory.register_parser(TXTParser)
        """
        parser = parser_class()
        for ext in parser.get_supported_extensions():
            cls._parsers[ext] = parser

    @classmethod
    def get_parser(cls, filename: str) -> BaseDocumentParser | None:
        """
        Возвращает подходящий парсер для указанного файла.

        Определяет расширение файла и ищет зарегистрированный парсер.
        Поиск выполняется в два этапа:
        1. Проверка точного совпадения расширения (например, '.pdf').
        2. Поиск по окончанию имени файла (например, 'data.jsonl').

        Это позволяет корректно обрабатывать составные расширения.

        Args:
            filename (str): Имя файла (может включать путь).

        Returns:
            BaseDocumentParser | None: Экземпляр подходящего парсера
            или None, если поддержка формата отсутствует.

        Example:
            >>> parser = DocumentParserFactory.get_parser("document.pdf")
            >>> if parser:
            ...     pages = parser.parse(file_bytes, "document.pdf")
        """
        ext = BaseDocumentParser.get_file_extension(filename)

        if ext in cls._parsers:
            return cls._parsers[ext]

        for registered_ext in cls._parsers:
            if filename.lower().endswith(registered_ext):
                return cls._parsers[registered_ext]

        return None

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит файл, используя подходящий парсер, выбранный автоматически.

        Основной метод для внешнего использования. Находит парсер для файла
        по его расширению и выполняет парсинг. Обрабатывает ошибки парсинга
        и выводит сообщения об ошибках через Streamlit.

        Args:
            file_bytes (bytes): Бинарные данные файла для парсинга.
            filename (str): Имя файла (для логирования и определения парсера).

        Returns:
            List[Tuple[int, str]]: Список кортежей (номер_фрагмента, текст_фрагмента),
            полученный от парсера, или пустой список в случае ошибки.

        Notes:
            - Если формат файла не поддерживается, выводится ошибка через st.error().
            - Любые исключения при парсинге перехватываются и не прерывают работу.

        Example:
            >>> with open("data.pdf", "rb") as f:
            ...     content = f.read()
            >>> pages = DocumentParserFactory.parse_file(content, "data.pdf")
            >>> for page_num, text in pages:
            ...     print(f"{page_num}: {text[:100]}...")
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


# ====================== Регистрация всех парсеров ======================
DocumentParserFactory.register_parser(PDFParser)
DocumentParserFactory.register_parser(TXTParser)
DocumentParserFactory.register_parser(CSVParser)
DocumentParserFactory.register_parser(JSONParser)   # ← добавили JSON


def parse_document(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная функция для парсинга документа.

    Обёртка над DocumentParserFactory.parse_file(), предоставляет
    простой интерфейс для использования в других модулях.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла.

    Returns:
        List[Tuple[int, str]]: Список текстовых фрагментов документа.

    Example:
        >>> content = b"Пример текста"
        >>> chunks = parse_document(content, "test.txt")
    """
    return DocumentParserFactory.parse_file(file_bytes, filename)


def get_supported_extensions() -> List[str]:
    """
    Возвращает список всех поддерживаемых расширений файлов.

    Собирает и возвращает все расширения, зарегистрированные в фабрике.
    Результат отсортирован по алфавиту.

    Returns:
        List[str]: Отсортированный список поддерживаемых расширений (например, ['.csv', '.json', '.pdf']).

    Example:
        >>> extensions = get_supported_extensions()
        >>> print("Поддерживаемые форматы:", ", ".join(extensions))
        Поддерживаемые форматы: .csv, .json, .pdf, .txt
    """
    exts = set()
    for parser in DocumentParserFactory._parsers.values():
        exts.update(parser.get_supported_extensions())
    return sorted(exts)