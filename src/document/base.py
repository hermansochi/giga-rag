"""
src/document/base.py

Базовый абстрактный интерфейс для всех парсеров документов.
Определяет единый контракт для обработки различных форматов файлов.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path


class BaseDocumentParser(ABC):
    """
    Абстрактный базовый класс для парсеров документов.

    Все конкретные парсеры (PDF, TXT, CSV, JSON и др.) должны наследовать
    этот класс и реализовывать метод parse().
    """

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит документ и возвращает список текстовых фрагментов.

        Args:
            file_bytes (bytes): Бинарные данные файла.
            filename (str): Имя файла (используется для логирования и определения типа).

        Returns:
            List[Tuple[int, str]]: Список кортежей (номер фрагмента, текст фрагмента).
        """
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Возвращает список поддерживаемых расширений файлов.

        Returns:
            List[str]: Список расширений в нижнем регистре с точкой.
        """
        return []

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Извлекает расширение файла в нижнем регистре с точкой.

        Args:
            filename (str): Имя файла (может включать путь).

        Returns:
            str: Расширение файла (например, '.pdf').
                 Возвращает пустую строку, если расширения нет.
        """
        return Path(filename).suffix.lower()