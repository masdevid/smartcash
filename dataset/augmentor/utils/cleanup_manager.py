"""
File: smartcash/dataset/augmentor/utils/cleanup_manager.py
Deskripsi: Manager untuk cleanup operations
"""

import shutil
from pathlib import Path
from typing import Dict, Any

class CleanupManager:
    """🧹 Manager untuk cleanup augmented data"""
    
    def __init__(self, config, progress_bridge=None):
        self.config = config
        self.progress = progress_bridge
        self.data_dir = config.get('data', {}).get('dir', 'data')
    
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """Cleanup augmented files dan symlinks"""
        try:
            total_removed = 0
            
            if target_split:
                splits = [target_split]
            else:
                splits = ['train', 'valid', 'test']
            
            for split in splits:
                # Remove augmented files
                aug_dir = Path(self.data_dir) / 'augmented' / split
                if aug_dir.exists():
                    removed = self._remove_augmented_files(aug_dir)
                    total_removed += removed
                
                # Remove symlinks dari preprocessed
                prep_dir = Path(self.data_dir) / 'preprocessed' / split
                if prep_dir.exists():
                    removed = self._remove_augmented_symlinks(prep_dir)
                    total_removed += removed
            
            return {
                'status': 'success',
                'total_removed': total_removed,
                'message': f'Removed {total_removed} augmented files and symlinks'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'total_removed': 0}
    
    def _remove_augmented_files(self, aug_dir: Path) -> int:
        """Remove augmented files dari directory"""
        count = 0
        
        for subdir in ['images', 'labels']:
            dir_path = aug_dir / subdir
            if dir_path.exists():
                for file_path in dir_path.glob('aug_*'):
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception:
                        continue
        
        return count
    
    def _remove_augmented_symlinks(self, prep_dir: Path) -> int:
        """Remove augmented symlinks dari preprocessed directory"""
        count = 0
        
        for subdir in ['images', 'labels']:
            dir_path = prep_dir / subdir
            if dir_path.exists():
                for file_path in dir_path.glob('aug_*'):
                    try:
                        if file_path.is_symlink():
                            file_path.unlink()
                            count += 1
                    except Exception:
                        continue
        