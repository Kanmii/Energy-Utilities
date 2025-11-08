"""HTML Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    print("BeautifulSoup not available. Install using: pip install beautifulsoup4")
    HTML_AVAILABLE = False

class HTMLProcessor(BaseProcessor):
    def extract_text(self, file_path: Path) -> Optional[str]:
        if not HTML_AVAILABLE:
            return "HTML processing not available - install beautifulsoup4"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            print(f"HTML processing error: {e}")
            return None

