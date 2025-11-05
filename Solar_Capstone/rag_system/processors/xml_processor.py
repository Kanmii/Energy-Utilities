"""XML Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

try:
 from bs4 import BeautifulSoup
 XML_AVAILABLE = True
except ImportError:
 XML_AVAILABLE = False

class XMLProcessor(BaseProcessor):
 def extract_text(self, file_path: Path) -> Optional[str]:
 if not XML_AVAILABLE:
 return "XML processing not available - install beautifulsoup4"
 
 try:
 with open(file_path, 'r', encoding='utf-8') as f:
 soup = BeautifulSoup(f.read(), 'xml')
 return soup.get_text()
 except Exception as e:
 print(f" XML processing error: {e}")
 return None

