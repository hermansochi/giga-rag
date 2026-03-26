# src/document/parsers/pdf_parser.py
"""
Парсер PDF-файлов с использованием PyMuPDF (fitz).
Извлекает текст с сохранением номеров страниц.
"""

from typing import List, Tuple
import fitz  # PyMuPDF
from ..base import BaseDocumentParser


class PDFParser(BaseDocumentParser):
    """
    Парсер для PDF документов.
    Поддерживает извлечение текста постранично.
    """

    def get_supported_extensions(self) -> List[str]:
        return [".pdf"]

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
        """
        Парсит PDF и возвращает список: [(page_number, page_text), ...]

        Args:
            file_bytes: байты PDF-файла
            filename: имя файла (для логирования)

        Returns:
            Список кортежей (номер страницы, текст страницы)
        """
        pages = []

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()

                # Пропускаем пустые страницы
                if text and len(text) > 10:
                    pages.append((page_num + 1, text))  # нумерация страниц с 1

            doc.close()

            if not pages:
                print(f"⚠️ В PDF {filename} не удалось извлечь текст")

            return pages

        except Exception as e:
            print(f"❌ Ошибка парсинга PDF {filename}: {e}")
            return []


# Для удобного импорта
def parse_pdf(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Удобная функция для использования извне"""
    parser = PDFParser()
    return parser.parse(file_bytes, filename)