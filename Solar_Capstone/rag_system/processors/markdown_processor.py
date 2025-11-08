"""Markdown Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

class MarkdownProcessor(BaseProcessor):
    def extract_text(self, file_path: Path) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Markdown processing error: {e}")
            return None

