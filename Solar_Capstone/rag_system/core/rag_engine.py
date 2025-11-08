"""
RAG Engine - Main orchestrator for document processing and retrieval
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from .document_processor import DocumentProcessor
from .mongodb_vector_store import MongoDBVectorStore
from .retrieval_system import RetrievalSystem

# Configure logging
logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG engine for document processing and retrieval"""
    
    def __init__(self, 
                mongo_url: str = "mongodb://localhost:27017", 
                db_name: str = "solar_recommender",
                knowledge_base_path: Optional[Union[str, Path]] = None):
        """Initialize RAG Engine
        
        Args:
            mongo_url: MongoDB connection URL
            db_name: Database name to use
            knowledge_base_path: Path to store knowledge base files
        """
        # Initialize components
        try:
            self.document_processor = DocumentProcessor()
            self.vector_store = MongoDBVectorStore(mongo_url, db_name)
            self.retrieval_system = RetrievalSystem(self.vector_store)
            
            # Set knowledge base path
            self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else Path.cwd() / "knowledge_base"
            self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("RAG Engine initialized with MongoDB")
            logger.info(f"MongoDB URL: {mongo_url}")
            logger.info(f"Database: {db_name}")
            logger.info(f"Knowledge base path: {self.knowledge_base_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {e}")
            raise
    
    def add_documents(self, source_path: Union[str, Path, List[Union[str, Path]]], 
                      recursive: bool = True, update_existing: bool = False) -> Dict[str, Any]:
        """Add documents to the knowledge base.

        Accepts a single path, a list of paths, or a directory. Processes files
        using the DocumentProcessor and stores embeddings via the vector store.
        """
        try:
            if isinstance(source_path, (str, Path)):
                source_paths = [Path(source_path)]
            else:
                source_paths = [Path(p) for p in source_path]

            processed_files: List[str] = []
            failed_files: List[str] = []
            skipped_files: List[str] = []
            errors: List[str] = []

            for source in source_paths:
                if source.is_file():
                    result = self._process_single_file(source, update_existing)
                    if result.get('success'):
                        processed_files.append(str(source))
                    else:
                        failed_files.append(str(source))
                        errors.append(result.get('error', 'unknown'))
                elif source.is_dir():
                    if recursive:
                        files = list(source.rglob("*"))
                    else:
                        files = list(source.iterdir())

                    for file_path in files:
                        if file_path.is_file():
                            result = self._process_single_file(file_path, update_existing)
                            if result.get('success'):
                                processed_files.append(str(file_path))
                            else:
                                failed_files.append(str(file_path))
                                errors.append(result.get('error', 'unknown'))

            return {
                "success": len(processed_files) > 0,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "skipped_files": skipped_files,
                "total_processed": len(processed_files),
                "total_failed": len(failed_files),
                "total_skipped": len(skipped_files),
                "processing_time": 0.0,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"add_documents failed: {e}")
            return {
                "success": False,
                "processed_files": [],
                "failed_files": [],
                "skipped_files": [],
                "total_processed": 0,
                "total_failed": 0,
                "total_skipped": 0,
                "processing_time": 0.0,
                "errors": [str(e)]
            }

    def _process_single_file(self, file_path: Path, update_existing: bool) -> Dict[str, Any]:
        """Process a single file and add it to the vector store."""
        try:
            if not self.document_processor.is_supported_file(file_path):
                return {"success": False, "error": f"Unsupported file type: {file_path.suffix}"}

            doc_id = str(file_path.absolute())
            if not update_existing and self.vector_store.document_exists(doc_id):
                return {"success": False, "error": "Document already exists"}

            document_data = self.document_processor.process_file(file_path)
            if not document_data:
                return {"success": False, "error": "Failed to process document"}

            success = self.vector_store.add_document(document_data)
            if not success:
                return {"success": False, "error": "Failed to add to vector store"}

            return {"success": True}

        except Exception as e:
            logger.error(f"_process_single_file error: {e}")
            return {"success": False, "error": str(e)}

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the knowledge base using the retrieval system."""
        try:
            return self.retrieval_system.search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            return {
                "total_documents": self.vector_store.get_document_count(),
                "file_types": self.vector_store.get_file_type_distribution(),
                "knowledge_base_size": self._get_knowledge_base_size(),
                "last_updated": datetime.now().isoformat(),
                "supported_extensions": len(self.document_processor.supported_extensions)
            }
        except Exception as e:
            logger.error(f"get_statistics error: {e}")
            return {"error": str(e)}

    def _get_knowledge_base_size(self) -> int:
        """Get knowledge base size in bytes"""
        try:
            total_size = 0
            for file_path in self.knowledge_base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0

    def clear_knowledge_base(self, confirm: bool = False) -> bool:
        """Clear the knowledge base"""
        if not confirm:
            return False

        try:
            self.vector_store.clear_all()
            for file_path in self.knowledge_base_path.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document"""
        try:
            return self.vector_store.get_document_info(document_id)
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return None
