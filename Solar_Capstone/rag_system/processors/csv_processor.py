"""CSV Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

try:
 import pandas as pd
 CSV_AVAILABLE = True
except ImportError:
 CSV_AVAILABLE = False

class CSVProcessor(BaseProcessor):
 def extract_text(self, file_path: Path) -> Optional[str]:
 if not CSV_AVAILABLE:
 return "CSV processing not available - install pandas"
 
 try:
 df = pd.read_csv(file_path)
 return df.to_string()
 except Exception as e:
 print(f" CSV processing error: {e}")
 return None

