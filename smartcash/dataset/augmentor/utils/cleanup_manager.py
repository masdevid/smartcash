"""
File: smartcash/dataset/augmentor/utils/cleanup_manager.py
Deskripsi: Manager untuk cleanup operations tanpa symlink handling
"""

import shutil
from pathlib import Path
from typing import Dict, Any

class CleanupManager:
    """🧹 Manager untuk cleanup augmented data tanpa symlinks"""
    
    def __init__(self, config, progress_bridge=None):
        self.config = config
        self.progress = progress_bridge
        self.data_dir = config.get('data', {}).get('dir', 'data')
    
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """Cleanup augmented files dan preprocessed files"""
        try:
            total_removed = 0
            
            if target_split:
                splits = [target_split]
            else:
                splits = ['train', 'valid', 'test']
            
            for split in splits:
                # Remove augmented files (.jpg dan .txt)
                aug_dir = Path(self.data_dir) / 'augmented' / split
                if aug_dir.exists():
                    removed = self._remove_augmented_files(aug_dir)
                    total_removed += removed
                
                # Remove preprocessed files (.npy, .jpg, .txt)
                prep_dir = Path(self.data_dir) / 'preprocessed' / split
                if prep_dir.exists():
                    removed = self._remove_preprocessed_files(prep_dir)
                    total_removed += removed
            
            return {
                'status': 'success',
                'total_removed': total_removed,
                'message': f'Dihapus {total_removed} file augmented dan preprocessed'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'total_removed': 0}
    
    def _remove_augmented_files(self, aug_dir: Path) -> int:
        """Remove augmented files dari directory"""
        count = 0
        
        # Remove dari images directory
        images_dir = aug_dir / 'images'
        if images_dir.exists():
            for file_path in images_dir.glob('aug_*.jpg'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        # Remove dari labels directory 
        labels_dir = aug_dir / 'labels'
        if labels_dir.exists():
            for file_path in labels_dir.glob('aug_*.txt'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        return count
    
    def _remove_preprocessed_files(self, prep_dir: Path) -> int:
        """Remove preprocessed files dari directory"""
        count = 0
        
        # Remove dari images directory (.npy dan .jpg files)
        images_dir = prep_dir / 'images'
        if images_dir.exists():
            # Remove .npy files (normalized untuk training)
            for file_path in images_dir.glob('aug_*.npy'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
            
            # Remove .jpg files (visualization)
            for file_path in images_dir.glob('aug_*.jpg'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        # Remove dari labels directory
        labels_dir = prep_dir / 'labels'
        if labels_dir.exists():
            for file_path in labels_dir.glob('aug_*.txt'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        return count