"""
File: smartcash/dataset/augmentor/utils/file_processor.py
Deskripsi: File processor untuk handling input files dengan format naming yang benar
"""

from pathlib import Path
from typing import List
import glob

class FileProcessor:
    """📁 Processor untuk handling input files dengan smart detection"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data', {}).get('dir', 'data')
    
    def get_split_files(self, split: str) -> List[str]:
        """Get files untuk specific split - prioritize preprocessed images over raw"""
        # First try preprocessed images (with padding applied)
        preprocessed_dir = Path(self.data_dir) / 'preprocessed' / split / 'images'
        
        if preprocessed_dir.exists():
            # Look for preprocessed image files (not .npy files)
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            files = []
            
            for ext in extensions:
                files.extend(glob.glob(str(preprocessed_dir / ext)))
            
            if files:
                return files
        
        # Fallback to raw images if preprocessed not available
        raw_dir = Path(self.data_dir) / split / 'images'
        
        if not raw_dir.exists():
            return []
        
        # Support multiple image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        
        for ext in extensions:
            files.extend(glob.glob(str(raw_dir / ext)))
        
        return files
    
    def get_label_path(self, image_path: str) -> str:
        """Get corresponding label path - handle both preprocessed and raw paths"""
        img_path = Path(image_path)
        
        # Check if this is a preprocessed image path
        if 'preprocessed' in img_path.parts:
            # For preprocessed images: data/preprocessed/split/images/pre_filename.jpg
            # Label should be: data/preprocessed/split/labels/pre_filename.txt
            label_dir = img_path.parent.parent / 'labels'
            return str(label_dir / f"{img_path.stem}.txt")
        else:
            # For raw images: data/split/images/filename.jpg  
            # Label should be: data/split/labels/filename.txt
            label_dir = img_path.parent.parent / 'labels'
            return str(label_dir / f"{img_path.stem}.txt")
