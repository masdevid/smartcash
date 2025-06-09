"""
File: smartcash/dataset/augmentor/utils/file_processor.py
Deskripsi: File processor untuk handling input files dengan format naming yang benar
"""

from pathlib import Path
from typing import List, Tuple
import glob

class FileProcessor:
    """📁 Processor untuk handling input files dengan smart detection"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data', {}).get('dir', 'data')
    
    def get_split_files(self, split: str) -> List[str]:
        """Get files untuk specific split"""
        split_dir = Path(self.data_dir) / split / 'images'
        
        if not split_dir.exists():
            return []
        
        # Support multiple image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        
        for ext in extensions:
            files.extend(glob.glob(str(split_dir / ext)))
        
        return files
    
    def get_label_path(self, image_path: str) -> str:
        """Get corresponding label path"""
        img_path = Path(image_path)
        label_dir = img_path.parent.parent / 'labels'
        return str(label_dir / f"{img_path.stem}.txt")
