"""JSON Processor"""
from pathlib import Path
from typing import Optional
import json
from .base_processor import BaseProcessor

class JSONProcessor(BaseProcessor):
 def extract_text(self, file_path: Path) -> Optional[str]:
 try:
 with open(file_path, 'r', encoding='utf-8') as f:
 data = json.load(f)
 return json.dumps(data, indent=2)
 except Exception as e:
 print(f" JSON processing error: {e}")
 return None

