# File: src/interfaces/handlers/cleaning_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk operasi pembersihan dataset

import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from interfaces.handlers.base_handler import BaseHandler    

class DataCleaningHandler(BaseHandler):
    """Handler for dataset cleaning operations"""
    def __init__(self, config):
        super().__init__(config)
        self.stats = {'removed': 0, 'errors': 0}
        
    def clean_dataset(self, mode: str) -> Dict:
        """Clean dataset based on specified mode"""
        self.logger.info(f"🧹 Memulai pembersihan dataset (mode: {mode})")
        self.stats = {'removed': 0, 'errors': 0}
        
        try:
            if mode == 'all':
                success = self._clean_all()
            elif mode == 'augmented':
                success = self._clean_augmented()
            elif mode == 'training':
                success = self._clean_training()
            elif mode == 'corrupt':
                success = self._clean_corrupt()
            else:
                self.logger.error(f"Mode pembersihan tidak valid: {mode}")
                return self.stats
                
            status = 'success' if success else 'failed'
            self.log_operation(f"Pembersihan ({mode})", status)
            
        except Exception as e:
            self.log_operation("Pembersihan", "failed", str(e))
            self.stats['errors'] += 1
            
        return self.stats
        
    def _clean_all(self) -> bool:
        """Remove all data except .gitkeep"""
        for img_dir, label_dir in self.get_split_dirs():
            for dir_path in [img_dir, label_dir]:
                if not dir_path.exists():
                    continue
                    
                for file_path in dir_path.glob('*'):
                    if file_path.name != '.gitkeep':
                        try:
                            file_path.unlink()
                            self.stats['removed'] += 1
                        except Exception as e:
                            self.logger.error(f"Gagal menghapus {file_path}: {str(e)}")
                            self.stats['errors'] += 1
                            
        return self.stats['errors'] == 0
        
    def _clean_augmented(self) -> bool:
        """Remove augmented data"""
        for img_dir, label_dir in self.get_split_dirs():
            for dir_path in [img_dir, label_dir]:
                if not dir_path.exists():
                    continue
                    
                for file_path in dir_path.glob('*_aug*'):
                    try:
                        file_path.unlink()
                        self.stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"Gagal menghapus {file_path}: {str(e)}")
                        self.stats['errors'] += 1
                        
        return self.stats['errors'] == 0
        
    def _clean_training(self) -> bool:
        """Remove training data"""
        train_img = self.rupiah_dir / 'train' / 'images'
        train_label = self.rupiah_dir / 'train' / 'labels'
        
        for dir_path in [train_img, train_label]:
            if not dir_path.exists():
                continue
                
            for file_path in dir_path.glob('*'):
                if file_path.name != '.gitkeep':
                    try:
                        file_path.unlink()
                        self.stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"Gagal menghapus {file_path}: {str(e)}")
                        self.stats['errors'] += 1
                        
        return self.stats['errors'] == 0
        
    def _clean_corrupt(self) -> bool:
        """Remove corrupt files"""
        success = True
        
        # Clean corrupt images
        success &= self._clean_corrupt_images()
        
        # Clean invalid labels
        success &= self._clean_invalid_labels()
        
        # Clean orphaned files
        success &= self._clean_orphaned_files()
        
        return success
        
    def _clean_corrupt_images(self) -> bool:
        """Clean corrupt image files"""
        for img_dir, _ in self.get_split_dirs():
            if not img_dir.exists():
                continue
                
            for img_path in img_dir.glob('*.jpg'):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        img_path.unlink()
                        self.stats['removed'] += 1
                except Exception as e:
                    self.logger.error(f"Gagal memproses {img_path}: {str(e)}")
                    self.stats['errors'] += 1
                    
        return self.stats['errors'] == 0
        
    def _clean_invalid_labels(self) -> bool:
        """Clean invalid label files"""
        for _, label_dir in self.get_split_dirs():
            if not label_dir.exists():
                continue
                
            for label_path in label_dir.glob('*.txt'):
                try:
                    if not self._validate_label_file(label_path):
                        label_path.unlink()
                        self.stats['removed'] += 1
                except Exception as e:
                    self.logger.error(f"Gagal memproses {label_path}: {str(e)}")
                    self.stats['errors'] += 1
                    
        return self.stats['errors'] == 0
        
    def _clean_orphaned_files(self) -> bool:
        """Clean files without corresponding pair"""
        for img_dir, label_dir in self.get_split_dirs():
            if not img_dir.exists() or not label_dir.exists():
                continue
                
            # Check orphaned labels
            for label_path in label_dir.glob('*.txt'):
                img_path = img_dir / f"{label_path.stem}.jpg"
                if not img_path.exists():
                    try:
                        label_path.unlink()
                        self.stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"Gagal menghapus {label_path}: {str(e)}")
                        self.stats['errors'] += 1
                        
            # Check orphaned images
            for img_path in img_dir.glob('*.jpg'):
                label_path = label_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    try:
                        img_path.unlink()
                        self.stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"Gagal menghapus {img_path}: {str(e)}")
                        self.stats['errors'] += 1
                        
        return self.stats['errors'] == 0
        
    def _validate_label_file(self, path: Path) -> bool:
        """Validate single label file"""
        try:
            with open(path) as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) != 5:  # class, x, y, w, h
                        return False
                    if not (0 <= values[0] <= 6):  # valid class
                        return False
                    if not all(0 <= v <= 1 for v in values[1:]):  # normalized coords
                        return False
            return True
        except Exception:
            return False