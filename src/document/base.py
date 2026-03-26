# src/document/base.py
"""
Базовый класс для всех парсеров документов.
Определяет общий интерфейс.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from pathlib import Path


class BaseDocumentParser(ABC):
    """
    Абстрактный базовый класс для парсеров документов.
    """

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит документ и возвращает список кортежей:
        [(page_number, page_text), ...]

        Для TXT-файлов page_number будет всегда 1.
        """
        pass


    def get_supported_extensions(self) -> List[str]:
        """Возвращает список поддерживаемых расширений (с точкой)"""
        return []


    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Возвращает расширение файла в нижнем регистре с точкой"""
        return Path(filename).suffix.lower()