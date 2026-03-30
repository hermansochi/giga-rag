"""
Модуль для парсинга текстовых файлов (TXT, MD, TEXT).

Содержит класс TXTParser, реализующий интерфейс BaseDocumentParser,
и вспомогательную функцию parse_txt().

Класс TXTParser поддерживает несколько кодировок и корректно обрабатывает
пустые файлы и ошибки декодирования.
"""

from typing import List, Tuple
from ..base import BaseDocumentParser
import streamlit as st


class TXTParser(BaseDocumentParser):
    """
    Парсер для текстовых файлов с расширениями .txt, .text, .md.

    Данный класс отвечает за декодирование бинарного содержимого текстового
    файла в строку UTF-8, обработку различных кодировок и возврат текста
    в структурированном виде.

    Поддерживаемые расширения файлов:
        - .txt
        - .text
        - .md (как обычный текст)

    Метод parse() пытается декодировать файл с использованием списка
    предопределённых кодировок. В случае неудачи применяется режим
    'replace' для обработки некорректных символов.

    Attributes:
        None (нет внутреннего состояния)

    Methods:
        get_supported_extensions() -> List[str]:
            Возвращает список поддерживаемых расширений файлов.

        parse(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
            Основной метод парсинга. Принимает бинарные данные файла
            и его имя. Возвращает список кортежей, где каждый кортеж
            содержит номер страницы (всегда 1 для TXT) и текст.
            Возвращает пустой список в случае ошибки или пустого файла.
    """

    def get_supported_extensions(self) -> List[str]:
        """
        Возвращает список расширений файлов, поддерживаемых этим парсером.

        Returns:
            List[str]: Список строк с расширениями, включая '.txt', '.text', '.md'.
        """
        return [".txt", ".text", ".md"]   # также поддерживаем .md как текст

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит бинарное содержимое TXT-файла и возвращает его как одну страницу.

        Метод пытается декодировать данные с использованием следующих кодировок
        в указанном порядке: utf-8, utf-8-sig, cp1251, koi8-r, latin1.
        При неудаче используется декодирование с заменой ошибок (errors='replace').

        Если после декодирования текст пустой, выводится предупреждение
        через Streamlit и возвращается пустой список.

        Args:
            file_bytes (bytes): Бинарные данные файла для парсинга.
            filename (str): Имя файла (используется для логирования ошибок).

        Returns:
            List[Tuple[int, str]]: Список с одним элементом [(1, текст)]
            или пустой список в случае ошибки или пустого файла.
        """
        try:
            # Пробуем разные кодировки
            text = None
            for encoding in ["utf-8", "utf-8-sig", "cp1251", "koi8-r", "latin1"]:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                # Если не удалось декодировать — используем замену ошибок
                text = file_bytes.decode("utf-8", errors="replace")

            text = text.strip()

            if not text:
                st.error(f"⚠️ TXT файл {filename} пустой")
                return []

            return [(1, text)]  # TXT считаем одной большой страницей

        except Exception as e:
            st.error(f"❌ Ошибка парсинга TXT {filename}: {e}")
            return []


# Удобная функция для внешнего использования
def parse_txt(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Удобная обёртка для парсинга TXT-файлов.

    Создаёт экземпляр TXTParser и вызывает его метод parse().

    Эта функция предоставляет простой интерфейс для использования
    парсера без необходимости явного создания объекта.

    Args:
        file_bytes (bytes): Бинарные данные файла.
        filename (str): Имя файла для логирования.

    Returns:
        List[Tuple[int, str]]: Результат парсинга — список с одной страницей
        или пустой список при ошибке.

    Example:
        >>> with open("example.txt", "rb") as f:
        ...     content = f.read()
        >>> pages = parse_txt(content, "example.txt")
        >>> if pages:
        ...     print(pages[0][1])  # печатает текст
    """
    parser = TXTParser()
    return parser.parse(file_bytes, filename)