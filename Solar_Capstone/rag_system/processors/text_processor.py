"""
Text Processor - Handles plain text files
"""

from pathlib import Path
from typing import Optional
import chardet

from .base_processor import BaseProcessor

class TextProcessor(BaseProcessor):
 """Processes plain text files"""
 
 def extract_text(self, file_path: Path) -> Optional[str]:
 """Extract text from file"""
 try:
 # Detect encoding
 with open(file_path, 'rb') as f:
 raw_data = f.read()
 encoding = chardet.detect(raw_data)['encoding']
 
 # Read with detected encoding
 with open(file_path, 'r', encoding=encoding) as f:
 return f.read()
 
 except Exception as e:
 print(f" Text processing error: {e}")
 return None

