# File: src/interfaces/handlers/cleaning_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk operasi pembersihan dataset dengan progress tracking

import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from interfaces.handlers.base_handler import BaseHandler

class DataCleaningHandler(BaseHandler):
    """Handler for dataset cleaning operations with progress tracking"""
    def __init__(self, config):
        super().__init__(config)
        self.stats = {'removed': 0, 'errors': 0}
        
    def clean_dataset(self, mode: str) -> Dict:
        """
        Clean dataset based on specified mode with progress tracking
        
        Args:
            mode: Mode pembersihan ('all', 'augmented', 'training', 'corrupt')
            
        Returns:
            Dict berisi statistik pembersihan
        """
        self.stats = {'removed': 0, 'errors': 0}
        
        try:
            # Validasi dan eksekusi mode pembersihan
            if mode == 'all':
                success = self._clean_all()
            elif mode == 'augmented':
                success = self._clean_augmented()
            elif mode == 'training':
                success = self._clean_training()
            elif mode == 'corrupt':
                success = self._clean_corrupt()
            else:
                self.logger.error(f"❌ Mode pembersihan tidak valid: {mode}")
                return self.stats
                
            # Log completion status
            status = 'success' if success else 'failed'
            self.log_operation(f"Pembersihan ({mode})", status)
            
        except Exception as e:
            self.log_operation("Pembersihan", "failed", str(e))
            self.stats['errors'] += 1
            
        return self.stats
        
    def _clean_all(self) -> bool:
        """Remove all data except .gitkeep"""
        total_files = sum(1 for _ in self._get_all_files())
        if total_files == 0:
            self.logger.info("🌵 Dataset sudah kosong!")
            return True
            
        with tqdm(total=total_files, desc="Menghapus semua data") as pbar:
            for file_path in self._get_all_files():
                if file_path.name != '.gitkeep':
                    try:
                        file_path.unlink()
                        self.stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Gagal menghapus {file_path}: {str(e)}")
                        self.stats['errors'] += 1
                pbar.update(1)
                    
        return self.stats['errors'] == 0
        
    def _clean_augmented(self) -> bool:
        """Remove augmented data with progress tracking"""
        aug_files = list(self._get_augmented_files())
        if not aug_files:
            self.logger.info("🎨 Tidak ada data augmentasi untuk dibersihkan!")
            return True
            
        with tqdm(total=len(aug_files), desc="Menghapus data augmentasi") as pbar:
            for file_path in aug_files:
                try:
                    file_path.unlink()
                    self.stats['removed'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal menghapus {file_path}: {str(e)}")
                    self.stats['errors'] += 1
                pbar.update(1)
                    
        return self.stats['errors'] == 0
        
    def _clean_training(self) -> bool:
        """Remove training data with progress tracking"""
        train_files = list(self._get_training_files())
        if not train_files:
            self.logger.info("🎓 Tidak ada data training untuk dibersihkan!")
            return True
            
        with tqdm(total=len(train_files), desc="Menghapus data training") as pbar:
            for file_path in train_files:
                if file_path.name != '.gitkeep':
                    try:
                        file_path.unlink()
                        self.stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Gagal menghapus {file_path}: {str(e)}")
                        self.stats['errors'] += 1
                pbar.update(1)
                    
        return self.stats['errors'] == 0
        
    def _clean_corrupt(self) -> bool:
        """Clean corrupt files with progress tracking"""
        success = True
        
        # Clean corrupt images
        success &= self._clean_corrupt_images()
        
        # Clean invalid labels
        success &= self._clean_invalid_labels()
        
        # Clean orphaned files
        success &= self._clean_orphaned_files()
        
        return success
        
    def _clean_corrupt_images(self) -> bool:
        """Clean corrupt image files with progress tracking"""
        img_files = list(self._get_image_files())
        if not img_files:
            self.logger.info("📸 Tidak ada gambar untuk diperiksa!")
            return True
            
        with tqdm(total=len(img_files), desc="Memeriksa gambar korup") as pbar:
            for img_path in img_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        img_path.unlink()
                        self.stats['removed'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal memproses {img_path}: {str(e)}")
                    self.stats['errors'] += 1
                pbar.update(1)
                    
        return self.stats['errors'] == 0
        
    def _clean_invalid_labels(self) -> bool:
        """Clean invalid label files with progress tracking"""
        label_files = list(self._get_label_files())
        if not label_files:
            self.logger.info("🏷️ Tidak ada label untuk diperiksa!")
            return True
            
        with tqdm(total=len(label_files), desc="Memeriksa label tidak valid") as pbar:
            for label_path in label_files:
                try:
                    if not self._validate_label_file(label_path):
                        label_path.unlink()
                        self.stats['removed'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal memproses {label_path}: {str(e)}")
                    self.stats['errors'] += 1
                pbar.update(1)
                    
        return self.stats['errors'] == 0
        
    def _clean_orphaned_files(self) -> bool:
        """Clean files without corresponding pair with progress tracking"""
        total_files = sum(1 for _ in self._get_all_files())
        if total_files == 0:
            self.logger.info("📂 Tidak ada file untuk diperiksa!")
            return True
            
        with tqdm(total=total_files, desc="Memeriksa file yatim piatu") as pbar:
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
                            self.logger.error(f"❌ Gagal menghapus {label_path}: {str(e)}")
                            self.stats['errors'] += 1
                    pbar.update(1)
                    
                # Check orphaned images
                for img_path in img_dir.glob('*.jpg'):
                    label_path = label_dir / f"{img_path.stem}.txt"
                    if not label_path.exists():
                        try:
                            img_path.unlink()
                            self.stats['removed'] += 1
                        except Exception as e:
                            self.logger.error(f"❌ Gagal menghapus {img_path}: {str(e)}")
                            self.stats['errors'] += 1
                    pbar.update(1)
                    
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
            
    def _get_all_files(self):
        """Generator for all dataset files"""
        for img_dir, label_dir in self.get_split_dirs():
            if img_dir.exists():
                yield from img_dir.glob('*')
            if label_dir.exists():
                yield from label_dir.glob('*')
                
    def _get_augmented_files(self):
        """Generator for augmented files"""
        for img_dir, label_dir in self.get_split_dirs():
            if img_dir.exists():
                yield from img_dir.glob('*_aug*')
            if label_dir.exists():
                yield from label_dir.glob('*_aug*')
                
    def _get_training_files(self):
        """Generator for training files"""
        train_img = self.rupiah_dir / 'train' / 'images'
        train_label = self.rupiah_dir / 'train' / 'labels'
        
        if train_img.exists():
            yield from train_img.glob('*')
        if train_label.exists():
            yield from train_label.glob('*')
            
    def _get_image_files(self):
        """Generator for image files"""
        for img_dir, _ in self.get_split_dirs():
            if img_dir.exists():
                yield from img_dir.glob('*.jpg')
                
    def _get_label_files(self):
        """Generator for label files"""
        for _, label_dir in self.get_split_dirs():
            if label_dir.exists():
                yield from label_dir.glob('*.txt')