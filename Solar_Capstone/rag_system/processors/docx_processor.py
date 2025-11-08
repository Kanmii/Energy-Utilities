"""DOCX Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    print("python-docx not available. Install using: pip install python-docx")
    DOCX_AVAILABLE = False

class DocxProcessor(BaseProcessor):
    def extract_text(self, file_path: Path) -> Optional[str]:
        if not DOCX_AVAILABLE:
            return "DOCX processing not available - install python-docx"
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"DOCX processing error: {e}")
            return None

