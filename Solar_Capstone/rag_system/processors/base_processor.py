"""
Base Processor - Abstract base class for document processors
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class BaseProcessor(ABC):
 """Abstract base class for document processors"""
 
 @abstractmethod
 def extract_text(self, file_path: Path) -> Optional[str]:
 """Extract text from file"""
 pass
 
 def is_supported(self, file_path: Path) -> bool:
 """Check if file is supported by this processor"""
 return True
 
 def get_metadata(self, file_path: Path) -> dict:
 """Get file metadata"""
 try:
 stat = file_path.stat()
 return {
 'size': stat.st_size,
 'modified': stat.st_mtime,
 'created': stat.st_ctime
 }
 except:
 return {}

