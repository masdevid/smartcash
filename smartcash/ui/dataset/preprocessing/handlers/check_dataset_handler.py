"""
File: smartcash/dataset/utils/path_validator.py
Deskripsi: Fixed path validator untuk dataset dengan mapping val->valid yang konsisten
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager


class DatasetPathValidator:
    """Validator untuk path dataset dengan mapping val->valid yang konsisten."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.env_manager = get_environment_manager()
        
        # Split mapping untuk konsistensi
        self.split_mapping = {
            'val': 'valid',
            'validation': 'valid'
        }
    
    def normalize_split_name(self, split: str) -> str:
        """Normalize split name untuk konsistensi."""
        return self.split_mapping.get(split.lower(), split.lower())
    
    def get_dataset_paths(self, base_dir: Optional[str] = None) -> Dict[str, str]:
        """Get dataset paths berdasarkan environment."""
        if base_dir:
            return {
                'data_root': base_dir,
                'train': f"{base_dir}/train",
                'valid': f"{base_dir}/valid", 
                'test': f"{base_dir}/test",
                'downloads': f"{base_dir}/downloads"
            }
        
        paths = get_paths_for_environment(
            self.env_manager.is_colab,
            self.env_manager.is_drive_mounted
        )
        return paths
    
    def detect_available_splits(self, data_dir: str) -> List[str]:
        """Deteksi splits yang tersedia dengan mapping val->valid."""
        data_path = Path(data_dir)
        available_splits = []
        
        # Check standard splits
        for split in ['train', 'valid', 'test']:
            split_dir = data_path / split
            if self._is_valid_split_dir(split_dir):
                available_splits.append(split)
        
        # Check legacy 'val' directory
        val_dir = data_path / 'val'
        if self._is_valid_split_dir(val_dir) and 'valid' not in available_splits:
            available_splits.append('valid')  # Normalize ke 'valid'
        
        return available_splits
    
    def _is_valid_split_dir(self, split_dir: Path) -> bool:
        """Check apakah split directory valid."""
        if not split_dir.exists():
            return False
            
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        return (images_dir.exists() and labels_dir.exists() and
                len(list(images_dir.glob('*.*'))) > 0)
    
    def get_split_path(self, data_dir: str, split: str) -> Path:
        """Get path untuk split dengan mapping val->valid."""
        data_path = Path(data_dir)
        normalized_split = self.normalize_split_name(split)
        
        # Check normalized path first
        split_path = data_path / normalized_split
        if split_path.exists():
            return split_path
        
        # Check legacy 'val' jika split adalah 'valid'
        if normalized_split == 'valid':
            val_path = data_path / 'val'
            if val_path.exists():
                return val_path
        
        return split_path
    
    def validate_dataset_structure(self, data_dir: str) -> Dict[str, any]:
        """Validate struktur dataset dengan detail info."""
        data_path = Path(data_dir)
        
        result = {
            'valid': data_path.exists(),
            'data_dir': str(data_path),
            'splits': {},
            'issues': [],
            'total_images': 0,
            'total_labels': 0
        }
        
        if not result['valid']:
            result['issues'].append(f"❌ Dataset directory tidak ditemukan: {data_dir}")
            return result
        
        # Check each split
        available_splits = self.detect_available_splits(data_dir)
        
        for split in ['train', 'valid', 'test']:
            split_path = self.get_split_path(data_dir, split)
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if split in available_splits:
                image_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                
                result['splits'][split] = {
                    'exists': True,
                    'path': str(split_path),
                    'images': image_count,
                    'labels': label_count,
                    'images_dir': str(images_dir),
                    'labels_dir': str(labels_dir)
                }
                
                result['total_images'] += image_count
                result['total_labels'] += label_count
                
                # Check issues
                if image_count == 0:
                    result['issues'].append(f"⚠️ Split {split}: Tidak ada gambar")
                if label_count == 0:
                    result['issues'].append(f"⚠️ Split {split}: Tidak ada label")
                if image_count != label_count:
                    result['issues'].append(f"⚠️ Split {split}: Gambar ({image_count}) ≠ Label ({label_count})")
            else:
                result['splits'][split] = {
                    'exists': False,
                    'path': str(split_path),
                    'images': 0,
                    'labels': 0
                }
                result['issues'].append(f"❌ Split {split}: Directory tidak ditemukan")
        
        return result
    
    def get_preprocessed_paths(self, base_dir: Optional[str] = None) -> Dict[str, str]:
        """Get preprocessed dataset paths."""
        paths = self.get_dataset_paths(base_dir)
        
        return {
            'preprocessed_root': f"{paths['data_root']}/preprocessed",
            'train': f"{paths['data_root']}/preprocessed/train",
            'valid': f"{paths['data_root']}/preprocessed/valid",
            'test': f"{paths['data_root']}/preprocessed/test"
        }
    
    def validate_preprocessed_structure(self, preprocessed_dir: str) -> Dict[str, any]:
        """Validate struktur preprocessed dataset."""
        preprocessed_path = Path(preprocessed_dir)
        
        result = {
            'valid': preprocessed_path.exists(),
            'preprocessed_dir': str(preprocessed_path),
            'splits': {},
            'total_processed': 0
        }
        
        if not result['valid']:
            return result
        
        # Check preprocessed splits
        for split in ['train', 'valid', 'test']:
            split_path = preprocessed_path / split
            
            if split_path.exists():
                processed_count = len(list(split_path.glob('**/*.jpg')))
                result['splits'][split] = {
                    'exists': True,
                    'path': str(split_path),
                    'processed': processed_count
                }
                result['total_processed'] += processed_count
            else:
                result['splits'][split] = {
                    'exists': False,
                    'path': str(split_path),
                    'processed': 0
                }
        
        return result


# Singleton instance
_path_validator = None

def get_path_validator(logger=None) -> DatasetPathValidator:
    """Get singleton path validator."""
    global _path_validator
    if _path_validator is None:
        _path_validator = DatasetPathValidator(logger)
    return _path_validator