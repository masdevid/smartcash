# File: src/handlers/dataset_handlers.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk operasi dataset SmartCash Detector

from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from utils.logging import ColoredLogger

class BaseHandler:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = ColoredLogger(self.__class__.__name__)

    def get_split_dirs(self) -> List[Tuple[Path, Path]]:
        """Get image and label directories for all splits"""
        splits = ['train', 'val', 'test']
        return [(self.data_dir / split / 'images', 
                self.data_dir / split / 'labels') 
                for split in splits]

class DatasetCopyHandler(BaseHandler):
    def copy_dataset(self, source_dir: Path) -> Dict:
        """Copy dataset from source to destination"""
        stats = {'copied': 0, 'skipped': 0, 'errors': 0}
        
        try:
            # Validate source structure
            if not self._validate_source(source_dir):
                return stats
            
            # Create destination structure
            self._create_dest_structure()
            
            # Copy files
            stats = self._copy_files(source_dir)
            
            # Create config
            self._create_config()
            
        except Exception as e:
            self.logger.error(f"Error copying dataset: {str(e)}")
            
        return stats

    def _validate_source(self, source_dir: Path) -> bool:
        """Validate source directory structure"""
        required_structure = {
            'train': ['images', 'labels'],
            'val': ['images', 'labels'],
            'test': ['images', 'labels']
        }
        
        for split, subdirs in required_structure.items():
            for subdir in subdirs:
                if not (source_dir / split / subdir).exists():
                    self.logger.error(f"Missing directory: {split}/{subdir}")
                    return False
        return True

    def _create_dest_structure(self):
        """Create destination directory structure"""
        for img_dir, label_dir in self.get_split_dirs():
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

    def _copy_files(self, source_dir: Path) -> Dict:
        """Copy files with progress tracking"""
        stats = {'copied': 0, 'skipped': 0, 'errors': 0}
        
        for img_dir, label_dir in self.get_split_dirs():
            src_img_dir = source_dir / img_dir.parent.name / 'images'
            src_label_dir = source_dir / label_dir.parent.name / 'labels'
            
            # Count files for progress bar
            total_files = len(list(src_img_dir.glob('*'))) + len(list(src_label_dir.glob('*')))
            
            with tqdm(total=total_files, desc=f"Copying {img_dir.parent.name}") as pbar:
                # Copy images
                for src_file in src_img_dir.glob('*'):
                    try:
                        shutil.copy2(src_file, img_dir / src_file.name)
                        stats['copied'] += 1
                    except Exception as e:
                        self.logger.error(f"Error copying {src_file}: {str(e)}")
                        stats['errors'] += 1
                    pbar.update(1)
                
                # Copy labels
                for src_file in src_label_dir.glob('*'):
                    try:
                        shutil.copy2(src_file, label_dir / src_file.name)
                        stats['copied'] += 1
                    except Exception as e:
                        self.logger.error(f"Error copying {src_file}: {str(e)}")
                        stats['errors'] += 1
                    pbar.update(1)
        
        return stats

class DatasetCleanHandler(BaseHandler):
    def clean_dataset(self, mode: str) -> Dict:
        """Clean dataset based on specified mode"""
        stats = {'removed': 0, 'errors': 0}
        
        try:
            if mode == 'all':
                stats = self._clean_all()
            elif mode == 'augmented':
                stats = self._clean_augmented()
            elif mode == 'training':
                stats = self._clean_training()
            elif mode == 'corrupt':
                stats = self._clean_corrupt()
        except Exception as e:
            self.logger.error(f"Error cleaning dataset: {str(e)}")
            stats['errors'] += 1
            
        return stats

    def _clean_all(self) -> Dict:
        """Remove all data except .gitkeep"""
        stats = {'removed': 0, 'errors': 0}
        
        for img_dir, label_dir in self.get_split_dirs():
            for dir_path in [img_dir, label_dir]:
                for file_path in dir_path.glob('*'):
                    if file_path.name != '.gitkeep':
                        try:
                            file_path.unlink()
                            stats['removed'] += 1
                        except Exception as e:
                            self.logger.error(f"Error removing {file_path}: {str(e)}")
                            stats['errors'] += 1
        
        return stats

    def _clean_augmented(self) -> Dict:
        """Remove augmented data"""
        stats = {'removed': 0, 'errors': 0}
        
        for img_dir, label_dir in self.get_split_dirs():
            for dir_path in [img_dir, label_dir]:
                for file_path in dir_path.glob('*_aug*'):
                    try:
                        file_path.unlink()
                        stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"Error removing {file_path}: {str(e)}")
                        stats['errors'] += 1
        
        return stats

    def _clean_training(self) -> Dict:
        """Remove training data"""
        stats = {'removed': 0, 'errors': 0}
        
        img_dir = self.data_dir / 'train' / 'images'
        label_dir = self.data_dir / 'train' / 'labels'
        
        for dir_path in [img_dir, label_dir]:
            for file_path in dir_path.glob('*'):
                if file_path.name != '.gitkeep':
                    try:
                        file_path.unlink()
                        stats['removed'] += 1
                    except Exception as e:
                        self.logger.error(f"Error removing {file_path}: {str(e)}")
                        stats['errors'] += 1
        
        return stats

    def _clean_corrupt(self) -> Dict:
        """Remove corrupt files"""
        stats = {'removed': 0, 'errors': 0}
        
        for img_dir, label_dir in self.get_split_dirs():
            # Check corrupt images
            for img_path in img_dir.glob('*.jpg'):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        img_path.unlink()
                        stats['removed'] += 1
                        
                        # Remove corresponding label
                        label_path = label_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            label_path.unlink()
                            stats['removed'] += 1
                except Exception:
                    stats['errors'] += 1
            
            # Check orphaned labels
            for label_path in label_dir.glob('*.txt'):
                img_path = img_dir / f"{label_path.stem}.jpg"
                if not img_path.exists():
                    try:
                        label_path.unlink()
                        stats['removed'] += 1
                    except Exception:
                        stats['errors'] += 1
        
        return stats

class DatasetVerifyHandler(BaseHandler):
    def verify_dataset(self) -> Dict:
        """Verify dataset integrity"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            stats[split] = self._verify_split(split)
        
        return stats

    def _verify_split(self, split: str) -> Dict:
        """Verify single dataset split"""
        img_dir = self.data_dir / split / 'images'
        label_dir = self.data_dir / split / 'labels'
        
        stats = {
            'images': 0,
            'labels': 0,
            'corrupt': 0,
            'invalid': 0,
            'augmented': 0,
            'original': 0
        }
        
        # Count and validate images
        for img_path in img_dir.glob('*.jpg'):
            stats['images'] += 1
            if '_aug' in img_path.stem:
                stats['augmented'] += 1
            else:
                stats['original'] += 1
                
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    stats['corrupt'] += 1
            except Exception:
                stats['corrupt'] += 1
        
        # Count and validate labels
        for label_path in label_dir.glob('*.txt'):
            stats['labels'] += 1
            try:
                with open(label_path) as f:
                    if not all(self._validate_label_line(line) for line in f):
                        stats['invalid'] += 1
            except Exception:
                stats['invalid'] += 1
        
        return stats

    def _validate_label_line(self, line: str) -> bool:
        """Validate single label line"""
        try:
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