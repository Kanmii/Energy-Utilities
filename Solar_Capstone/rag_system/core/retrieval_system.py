"""
Retrieval System - Advanced search and filtering for RAG system
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import re

from .vector_store import VectorStore

class RetrievalSystem:
 """Advanced retrieval system with filtering and ranking"""
 
 def __init__(self, vector_store: VectorStore):
 self.vector_store = vector_store
 
 def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7,
 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
 """Advanced search with filtering"""
 try:
 # Get basic search results
 results = self.vector_store.search_similar(
 query=query,
 top_k=top_k * 2, # Get more results for filtering
 similarity_threshold=similarity_threshold
 )
 
 # Apply filters if provided
 if filters:
 results = self._apply_filters(results, filters)
 
 # Apply ranking
 results = self._rank_results(results, query)
 
 # Return top_k results
 return results[:top_k]
 
 except Exception as e:
 print(f" Retrieval system error: {e}")
 return []
 
 def _apply_filters(self, results: List[Dict[str, Any]], 
 filters: Dict[str, Any]) -> List[Dict[str, Any]]:
 """Apply filters to search results"""
 filtered_results = []
 
 for result in results:
 if self._matches_filters(result, filters):
 filtered_results.append(result)
 
 return filtered_results
 
 def _matches_filters(self, result: Dict[str, Any], 
 filters: Dict[str, Any]) -> bool:
 """Check if result matches all filters"""
 try:
 doc_metadata = result.get('document_metadata', {})
 
 # File type filter
 if 'file_types' in filters:
 file_ext = doc_metadata.get('file_extension', '').lower()
 if file_ext not in [ft.lower() for ft in filters['file_types']]:
 return False
 
 # Date range filter
 if 'date_range' in filters:
 date_range = filters['date_range']
 added_at = doc_metadata.get('added_at', '')
 if added_at:
 doc_date = datetime.fromisoformat(added_at.replace('Z', '+00:00'))
 if 'start' in date_range and doc_date < date_range['start']:
 return False
 if 'end' in date_range and doc_date > date_range['end']:
 return False
 
 # File size filter
 if 'file_size_range' in filters:
 size_range = filters['file_size_range']
 file_size = doc_metadata.get('file_size', 0)
 if 'min' in size_range and file_size < size_range['min']:
 return False
 if 'max' in size_range and file_size > size_range['max']:
 return False
 
 # Content type filter
 if 'content_types' in filters:
 # Simple content type detection
 content_type = self._detect_content_type(result['chunk_text'])
 if content_type not in filters['content_types']:
 return False
 
 return True
 
 except Exception as e:
 print(f" Filter matching error: {e}")
 return True # Include result if filter check fails
 
 def _detect_content_type(self, text: str) -> str:
 """Detect content type from text"""
 text_lower = text.lower()
 
 if any(word in text_lower for word in ['technical', 'specification', 'manual', 'guide']):
 return 'technical'
 elif any(word in text_lower for word in ['educational', 'tutorial', 'learn', 'course']):
 return 'educational'
 elif any(word in text_lower for word in ['news', 'article', 'blog', 'report']):
 return 'news'
 elif any(word in text_lower for word in ['code', 'function', 'class', 'import']):
 return 'code'
 else:
 return 'document'
 
 def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
 """Rank results by relevance"""
 try:
 query_terms = set(query.lower().split())
 
 for result in results:
 # Calculate additional relevance score
 relevance_score = self._calculate_relevance_score(result, query_terms)
 result['relevance_score'] = relevance_score
 
 # Combine similarity and relevance
 result['combined_score'] = (
 result['similarity_score'] * 0.7 + 
 relevance_score * 0.3
 )
 
 # Sort by combined score
 results.sort(key=lambda x: x['combined_score'], reverse=True)
 return results
 
 except Exception as e:
 print(f" Ranking error: {e}")
 return results
 
 def _calculate_relevance_score(self, result: Dict[str, Any], 
 query_terms: set) -> float:
 """Calculate relevance score based on query terms"""
 try:
 chunk_text = result['chunk_text'].lower()
 chunk_terms = set(chunk_text.split())
 
 # Term overlap
 overlap = len(query_terms.intersection(chunk_terms))
 term_score = overlap / len(query_terms) if query_terms else 0
 
 # Title/heading boost
 title_boost = 0
 if any(term in chunk_text[:100] for term in query_terms):
 title_boost = 0.2
 
 # Length penalty (prefer medium-length chunks)
 length_penalty = 0
 chunk_length = len(chunk_text.split())
 if chunk_length < 10:
 length_penalty = -0.1
 elif chunk_length > 500:
 length_penalty = -0.05
 
 return min(1.0, term_score + title_boost + length_penalty)
 
 except Exception as e:
 print(f" Relevance calculation error: {e}")
 return 0.0
 
 def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
 """Get summary of a document"""
 try:
 doc_info = self.vector_store.get_document_info(document_id)
 if not doc_info:
 return None
 
 return {
 'document_id': document_id,
 'file_name': doc_info['file_name'],
 'file_path': doc_info['file_path'],
 'file_size': doc_info['file_size'],
 'file_extension': doc_info['file_extension'],
 'processed_at': doc_info['metadata'].get('processed_at'),
 'chunk_count': len(doc_info['chunks']),
 'added_at': doc_info['added_at']
 }
 
 except Exception as e:
 print(f" Document summary error: {e}")
 return None
 
 def get_related_documents(self, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
 """Get documents related to a specific document"""
 try:
 doc_info = self.vector_store.get_document_info(document_id)
 if not doc_info:
 return []
 
 # Use first chunk as query
 query = doc_info['chunks'][0] if doc_info['chunks'] else ""
 if not query:
 return []
 
 # Search for similar documents
 results = self.vector_store.search_similar(
 query=query,
 top_k=top_k + 1, # +1 to exclude the original document
 similarity_threshold=0.5
 )
 
 # Filter out the original document
 related_results = [
 result for result in results 
 if result['document_id'] != document_id
 ]
 
 return related_results[:top_k]
 
 except Exception as e:
 print(f" Related documents error: {e}")
 return []

