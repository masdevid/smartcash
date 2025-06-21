"""
File: smartcash/dataset/augmentor/utils/file_scanner.py
Deskripsi: Scanner untuk augmented files
"""
from pathlib import Path
from typing import List
import glob

class FileScanner:
    """🔍 Scanner untuk augmented files dengan pattern matching"""
    
    def scan_augmented_files(self, aug_path: str) -> List[str]:
        """Scan augmented files dengan pattern aug_*"""
        aug_dir = Path(aug_path) / 'images'
        
        if not aug_dir.exists():
            return []
        
        return list(glob.glob(str(aug_dir / 'aug_*.jpg')))

    def scan_preprocessed_files(self, prep_path: str) -> List[str]:
        """Scan preprocessed files dengan pattern pre_*"""
        prep_dir = Path(prep_path) / 'images'
        
        if not prep_dir.exists():
            return []
        
        return list(glob.glob(str(prep_path / 'pre_*.jpg')))