"""
Document Processor - Handles various file formats for RAG system
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
from datetime import datetime

# Import processors
from ..processors.text_processor import TextProcessor
from ..processors.pdf_processor import PDFProcessor
from ..processors.docx_processor import DocxProcessor
from ..processors.html_processor import HTMLProcessor
from ..processors.csv_processor import CSVProcessor
from ..processors.json_processor import JSONProcessor
from ..processors.xml_processor import XMLProcessor
from ..processors.markdown_processor import MarkdownProcessor
from ..processors.powerpoint_processor import PowerPointProcessor
from ..processors.excel_processor import ExcelProcessor

class DocumentProcessor:
 """Processes various document formats for RAG system"""
 
 def __init__(self):
 self.supported_extensions = [
 '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h',
 '.pdf', '.docx', '.html', '.htm', '.csv', '.json', '.xml',
 '.yaml', '.yml', '.pptx', '.xlsx', '.xls'
 ]
 
 # Initialize processors
 self.processors = {
 '.txt': TextProcessor(),
 '.md': MarkdownProcessor(),
 '.py': TextProcessor(),
 '.js': TextProcessor(),
 '.java': TextProcessor(),
 '.cpp': TextProcessor(),
 '.c': TextProcessor(),
 '.h': TextProcessor(),
 '.pdf': PDFProcessor(),
 '.docx': DocxProcessor(),
 '.html': HTMLProcessor(),
 '.htm': HTMLProcessor(),
 '.csv': CSVProcessor(),
 '.json': JSONProcessor(),
 '.xml': XMLProcessor(),
 '.yaml': TextProcessor(),
 '.yml': TextProcessor(),
 '.pptx': PowerPointProcessor(),
 '.xlsx': ExcelProcessor(),
 '.xls': ExcelProcessor()
 }
 
 print(f" Document Processor initialized with {len(self.processors)} processors")
 
 def is_supported_file(self, file_path: Path) -> bool:
 """Check if file is supported"""
 return file_path.suffix.lower() in self.processors
 
 def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
 """Process a single file"""
 try:
 if not self.is_supported_file(file_path):
 print(f" Unsupported file type: {file_path.suffix}")
 return None
 
 # Get processor for file type
 processor = self.processors.get(file_path.suffix.lower())
 if not processor:
 print(f" No processor available for: {file_path.suffix}")
 return None
 
 # Process the file
 content = processor.extract_text(file_path)
 if not content:
 print(f" Failed to extract content from: {file_path}")
 return None
 
 # Create document data
 document_data = {
 "document_id": str(file_path.absolute()),
 "file_name": file_path.name,
 "file_path": str(file_path),
 "file_size": file_path.stat().st_size,
 "file_extension": file_path.suffix,
 "content": content,
 "chunks": self._chunk_text(content),
 "metadata": {
 "processed_at": datetime.now().isoformat(),
 "file_hash": self._calculate_file_hash(file_path)
 },
 "additional_info": {
 "word_count": len(content.split()),
 "character_count": len(content),
 "chunk_count": len(self._chunk_text(content))
 }
 }
 
 return document_data
 
 except Exception as e:
 print(f" Error processing {file_path}: {e}")
 return None
 
 def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
 """Split text into chunks"""
 if len(text) <= chunk_size:
 return [text]
 
 chunks = []
 start = 0
 
 while start < len(text):
 end = start + chunk_size
 
 # Try to break at sentence boundary
 if end < len(text):
 # Look for sentence endings
 for i in range(end, max(start, end - 100), -1):
 if text[i] in '.!?':
 end = i + 1
 break
 
 chunk = text[start:end].strip()
 if chunk:
 chunks.append(chunk)
 
 start = end - overlap
 if start >= len(text):
 break
 
 return chunks
 
 def _calculate_file_hash(self, file_path: Path) -> str:
 """Calculate file hash for change detection"""
 try:
 with open(file_path, 'rb') as f:
 return hashlib.md5(f.read()).hexdigest()
 except:
 return ""
 
 def get_supported_extensions(self) -> List[str]:
 """Get list of supported file extensions"""
 return self.supported_extensions.copy()
 
 def validate_file(self, file_path: Path) -> Dict[str, Any]:
 """Validate if file can be processed"""
 return {
 "valid": file_path.exists() and file_path.is_file(),
 "supported": self.is_supported_file(file_path),
 "exists": file_path.exists(),
 "readable": file_path.is_file() and os.access(file_path, os.R_OK),
 "processor_available": file_path.suffix.lower() in self.processors,
 "errors": []
 }

