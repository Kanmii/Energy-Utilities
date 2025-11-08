"""
MongoDB Vector Store - Native vector search with MongoDB
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import pymongo
from pymongo import MongoClient

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using basic text matching")

class MongoDBVectorStore:
    """MongoDB-based vector store with native vector search"""

    def __init__(self, mongo_url: str = "mongodb://localhost:27017", db_name: str = "solar_recommender"):
        # MongoDB connection
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db.vector_embeddings

        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dimension = 384
                print("Sentence Transformers model loaded")
            except Exception as e:
                print(f"Failed to load Sentence Transformers: {e}")
                self.embedding_model = None
                self.embedding_dimension = 0
        else:
            self.embedding_model = None
            self.embedding_dimension = 0

        # Create vector search index if it doesn't exist
        self._create_vector_index()

        print(f"MongoDB Vector Store initialized")
        print(f"Database: {db_name}")
        print(f"Collection: vector_embeddings")
        try:
            print(f"Documents: {self.collection.count_documents({})}")
        except Exception:
            print("Documents: N/A (unable to count documents)")

    def _create_vector_index(self):
        """Create vector search index in MongoDB (if supported)."""
        try:
            # Check if collection exists
            existing_collections = self.db.list_collection_names()
            if "vector_embeddings" not in existing_collections:
                # Collection doesn't exist yet; nothing to index
                return

            # Try to create vector search index (MongoDB Atlas / enterprise feature)
            try:
                self.db.vector_embeddings.create_index([
                    ("embedding", "vectorSearch")
                ], {
                    "name": "vector_search_index",
                    "vectorSearch": {
                        "dimensions": self.embedding_dimension,
                        "similarity": "cosine"
                    }
                })
                print("Vector search index created")
            except Exception as e:
                # Index creation may not be supported on all MongoDB editions
                print(f"ℹ Vector search index may already exist or could not be created: {e}")
        except Exception as e:
            print(f"Could not create vector search index: {e}")

    def add_document(self, document_data: Dict[str, Any]) -> bool:
        """Add document chunks and embeddings to MongoDB."""
        try:
            document_id = document_data['document_id']

            # Generate embeddings for chunks
            chunks = document_data.get('chunks', [])
            if not chunks:
                print(f"No chunks found for document: {document_id}")
                return False

            # Create embeddings
            if self.embedding_model:
                chunk_embeddings = self.embedding_model.encode(chunks)
            else:
                # Fallback: use simple text features
                chunk_embeddings = self._create_simple_embeddings(chunks)

            # Store each chunk as a separate document in MongoDB
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                doc = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "file_name": document_data.get('file_name'),
                    "file_path": document_data.get('file_path'),
                    "file_size": document_data.get('file_size'),
                    "file_extension": document_data.get('file_extension'),
                    "metadata": document_data.get('metadata', {}),
                    "additional_info": document_data.get('additional_info', {}),
                    "added_at": datetime.now().isoformat()
                }

                # Insert or update chunk
                self.collection.replace_one(
                    {"document_id": document_id, "chunk_index": i},
                    doc,
                    upsert=True
                )

            print(f"Document added to MongoDB: {document_data.get('file_name')}")
            return True

        except Exception as e:
            print(f"Failed to add document to MongoDB: {e}")
            return False

    def _create_simple_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create simple embeddings when sentence-transformers is not available."""
        # Simple TF-IDF-like features
        embeddings: List[List[float]] = []
        for chunk in chunks:
            features = [
                len(chunk),
                len(chunk.split()),
                chunk.count(' '),
                chunk.count('\n'),
                sum(1 for c in chunk if c.isalpha()),
                sum(1 for c in chunk if c.isdigit()),
                sum(1 for c in chunk if c.isspace())
            ]
            embeddings.append(features)

        return np.array(embeddings)

    def search_similar(self, query: str, top_k: int = 5,
                       similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents using MongoDB vector search (or fallback)."""
        try:
            if not self.embedding_model:
                # Fallback to text search
                return self._text_search(query, top_k)

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Use MongoDB vector search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_search_index",
                        "path": "embedding",
                        "queryVector": query_embedding[0].tolist(),
                        "numCandidates": top_k * 2,
                        "limit": top_k
                    }
                },
                {
                    "$project": {
                        "document_id": 1,
                        "chunk_index": 1,
                        "chunk_text": 1,
                        "file_name": 1,
                        "file_path": 1,
                        "file_extension": 1,
                        "metadata": 1,
                        "added_at": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))

            # Format results
            formatted_results: List[Dict[str, Any]] = []
            for result in results:
                if result.get('score', 0) >= similarity_threshold:
                    formatted_results.append({
                        'document_id': result['document_id'],
                        'similarity_score': float(result['score']),
                        'chunk_index': result['chunk_index'],
                        'chunk_text': result['chunk_text'],
                        'document_metadata': {
                            'file_name': result.get('file_name'),
                            'file_path': result.get('file_path'),
                            'file_extension': result.get('file_extension'),
                            'metadata': result.get('metadata', {}),
                            'added_at': result.get('added_at')
                        },
                        'chunk_count': 1,  # Each chunk is a separate document
                        'search_metadata': {
                            'query': query,
                            'similarity_threshold': similarity_threshold,
                            'search_time': datetime.now().isoformat()
                        }
                    })

            return formatted_results[:top_k]

        except Exception as e:
            print(f"MongoDB vector search error: {e}")
            # Fallback to text search
            return self._text_search(query, top_k)

    def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback text search when vector search is not available."""
        try:
            # Simple text search using MongoDB text index
            results = self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)

            formatted_results: List[Dict[str, Any]] = []
            for result in results:
                formatted_results.append({
                    'document_id': result['document_id'],
                    'similarity_score': float(result.get('score', 0)),
                    'chunk_index': result.get('chunk_index'),
                    'chunk_text': result.get('chunk_text'),
                    'document_metadata': {
                        'file_name': result.get('file_name'),
                        'file_path': result.get('file_path'),
                        'file_extension': result.get('file_extension'),
                        'metadata': result.get('metadata', {}),
                        'added_at': result.get('added_at')
                    },
                    'chunk_count': 1,
                    'search_metadata': {
                        'query': query,
                        'search_time': datetime.now().isoformat()
                    }
                })

            return formatted_results

        except Exception as e:
            print(f"Text search error: {e}")
            return []

    def document_exists(self, document_id: str) -> bool:
        """Check if document exists in MongoDB"""
        return self.collection.count_documents({"document_id": document_id}) > 0

    def get_document_count(self) -> int:
        """Get total number of documents"""
        return self.collection.count_documents({})

    def get_file_type_distribution(self) -> Dict[str, int]:
        """Get distribution of file types"""
        pipeline = [
            {"$group": {"_id": "$file_extension", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        distribution: Dict[str, int] = {}
        for result in self.collection.aggregate(pipeline):
            distribution[result['_id']] = result['count']

        return distribution

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document"""
        try:
            # Get all chunks for the document
            chunks = list(self.collection.find({"document_id": document_id}))
            if not chunks:
                return None

            # Get first chunk for metadata
            first_chunk = chunks[0]

            return {
                'document_id': document_id,
                'file_name': first_chunk.get('file_name'),
                'file_path': first_chunk.get('file_path'),
                'file_size': first_chunk.get('file_size'),
                'file_extension': first_chunk.get('file_extension'),
                'chunks': [chunk.get('chunk_text') for chunk in chunks],
                'metadata': first_chunk.get('metadata', {}),
                'additional_info': first_chunk.get('additional_info', {}),
                'added_at': first_chunk.get('added_at'),
                'chunk_count': len(chunks)
            }

        except Exception as e:
            print(f"Error getting document info: {e}")
            return None

    def clear_all(self):
        """Clear all embeddings and metadata"""
        self.collection.delete_many({})
        print("MongoDB vector store cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            total_chunks = self.collection.count_documents({})
            unique_documents = len(self.collection.distinct("document_id"))

            return {
                'total_embeddings': total_chunks,
                'total_documents': unique_documents,
                'embedding_dimension': self.embedding_dimension,
                'store_size_mb': self._get_store_size_mb(),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_store_size_mb(self) -> float:
        """Get store size in MB"""
        try:
            stats = self.db.command("collStats", "vector_embeddings")
            return stats.get('size', 0) / (1024 * 1024)
        except Exception:
            return 0.0

