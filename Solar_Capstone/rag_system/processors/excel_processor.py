"""Excel Processor"""
from pathlib import Path
from typing import Optional
from .base_processor import BaseProcessor

try:
    import pandas as pd
    import openpyxl  # Required for Excel support
    EXCEL_AVAILABLE = True
except ImportError:
    print("Excel support not available. Install using: pip install pandas openpyxl")
    EXCEL_AVAILABLE = False

class ExcelProcessor(BaseProcessor):
    def extract_text(self, file_path: Path) -> Optional[str]:
        if not EXCEL_AVAILABLE:
            return "Excel processing not available - install openpyxl"
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text = ""
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"Sheet: {sheet_name}\n"
                text += df.to_string() + "\n\n"
            return text
        except Exception as e:
            print(f"Excel processing error: {e}")
            return None

