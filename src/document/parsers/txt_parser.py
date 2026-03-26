# src/document/parsers/txt_parser.py
"""
Парсер текстовых (.txt) файлов.
Простой, но надёжный.
"""

from typing import List, Tuple
from ..base import BaseDocumentParser


class TXTParser(BaseDocumentParser):
    """
    Парсер для TXT файлов.
    Возвращает весь текст как страницу №1.
    """

    def get_supported_extensions(self) -> List[str]:
        return [".txt", ".text", ".md"]   # также поддерживаем .md как текст

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит TXT-файл и возвращает список с одной "страницей".

        Returns:
            [(1, весь_текст)]
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
                print(f"⚠️ TXT файл {filename} пустой")
                return []

            return [(1, text)]  # TXT считаем одной большой страницей

        except Exception as e:
            print(f"❌ Ошибка парсинга TXT {filename}: {e}")
            return []


# Удобная функция для внешнего использования
def parse_txt(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Удобная обёртка"""
    parser = TXTParser()
    return parser.parse(file_bytes, filename)