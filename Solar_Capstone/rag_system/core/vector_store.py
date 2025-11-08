"""Abstract VectorStore interface used by RetrievalSystem and RAG core"""
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add_document(self, document_data: Dict[str, Any]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def search_similar(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_document_count(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def clear_all(self) -> None:
        raise NotImplementedError()
