"""
src/models/dto.py

Data Transfer Objects — чистые структуры данных.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass(frozen=True)
class DocumentChunk:
    """Один чанк документа из векторной БД."""
    
    chunk_text: str
    filename: str
    chunk_index: int
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def short_text(self) -> str:
        """Короткая версия для источников (~350 символов)."""
        return self.chunk_text[:350] + "..." if len(self.chunk_text) > 350 else self.chunk_text

    @property
    def full_preview(self) -> str:
        """Средняя версия для блока контекста (~450 символов)."""
        return self.chunk_text[:450] + "..." if len(self.chunk_text) > 450 else self.chunk_text

    @classmethod
    def from_db_row(cls, row: Any) -> "DocumentChunk":
        """Создаёт DocumentChunk из строки БД."""
        if hasattr(row, 'keys') and callable(getattr(row, 'keys')):
            data = dict(row)
        else:
            data = {
                "chunk_text": row[0],
                "filename": row[1],
                "chunk_index": row[2],
                "distance": row[3],
            }
        
        return cls(
            chunk_text=str(data.get("chunk_text", "")),
            filename=str(data.get("filename", "неизвестный")),
            chunk_index=int(data.get("chunk_index", 0)),
            distance=float(data.get("distance", 0.0)),
            metadata=data.get("metadata") or {}
        )

@dataclass(frozen=True)
class RerankCandidate:
    """Кандидат для реранкинга."""
    
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "неизвестный")
    
    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)
    
    @classmethod
    def from_document_chunk(cls, chunk: DocumentChunk) -> "RerankCandidate":
        """Преобразует DocumentChunk в RerankCandidate."""
        return cls(
            text=chunk.chunk_text,
            metadata={
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "distance": chunk.distance,
                **chunk.metadata
            }
        )

@dataclass(frozen=True)
class RerankedResult:
    """Результат после реранкинга."""
    text: str
    metadata: Dict[str, Any]
    rerank_score: Optional[float] = None


@dataclass(frozen=True)
class ChatSource:
    """Источник для отображения в UI."""
    filename: str
    distance: float
    text_preview: str
    chunk_index: int


@dataclass(frozen=True)
class ChatResponse:
    """Финальный ответ для Streamlit."""
    answer: str
    sources: List[ChatSource]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    response_time: float
    model_name: str
    used_rag: bool = True
    reranker_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)