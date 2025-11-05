"""PDF Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

try:
 import PyPDF2
 PDF_AVAILABLE = True
except ImportError:
 PDF_AVAILABLE = False

class PDFProcessor(BaseProcessor):
 def extract_text(self, file_path: Path) -> Optional[str]:
 if not PDF_AVAILABLE:
 return "PDF processing not available - install PyPDF2"
 
 try:
 with open(file_path, 'rb') as f:
 reader = PyPDF2.PdfReader(f)
 text = ""
 for page in reader.pages:
 text += page.extract_text() + "\n"
 return text
 except Exception as e:
 print(f" PDF processing error: {e}")
 return None

