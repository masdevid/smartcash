"""
File: smartcash/dataset/augmentor/utils/cleanup_manager.py
Deskripsi: Manager cleanup tanpa .jpg di preprocessed
"""

import shutil
from pathlib import Path
from typing import Dict, Any

class CleanupManager:
    """🧹 Manager cleanup augmented data"""
    
    def __init__(self, config, progress_bridge=None):
        self.config = config
        self.progress = progress_bridge
        self.data_dir = config.get('data', {}).get('dir', 'data')
    
    def cleanup_data(self, target: str = 'both', target_split: str = None) -> Dict[str, Any]:
        """Cleanup data berdasarkan target yang ditentukan.
        
        Args:
            target: Jenis file yang akan dibersihkan. Pilihan: 'augmented', 'preprocessed', atau 'both'
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test'). 
                       Jika None, semua split akan dibersihkan.
                        
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        try:
            if target not in ['augmented', 'preprocessed', 'both']:
                return {
                    'status': 'error',
                    'total_removed': 0,
                    'message': "Parameter target harus salah satu dari: 'augmented', 'preprocessed', atau 'both'"
                }
                
            total_removed = 0
            splits = [target_split] if target_split else ['train', 'valid', 'test']
            cleaned_targets = []
            
            cleanup_actions = {
                'augmented': self._remove_augmented_files,
                'preprocessed': self._remove_preprocessed_files
            }
            
            for split in splits:
                for t in (['augmented', 'preprocessed'] if target == 'both' else [target]):
                    target_dir = Path(self.data_dir) / t / split
                    if target_dir.exists():
                        removed = cleanup_actions[t](target_dir)
                        if removed > 0:
                            total_removed += removed
                            cleaned_targets.append(f'{t}/{split}')
            
            if not cleaned_targets:
                message = "Tidak ada file yang perlu dibersihkan"
            else:
                target_desc = ' dan '.join(cleaned_targets)
                message = f'Dihapus {total_removed} file dari {target_desc}'
            
            return {
                'status': 'success',
                'total_removed': total_removed,
                'message': message,
                'cleaned_targets': cleaned_targets
            }
            
        except Exception as e:
            return {
                'status': 'error', 
                'message': f'Gagal membersihkan file: {str(e)}', 
                'total_removed': 0,
                'cleaned_targets': []
            }
    
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """Alias untuk membersihkan data augmented.
        
        Args:
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test').
                       Jika None, semua split akan dibersihkan.
                       
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        return self.cleanup_data(target='augmented', target_split=target_split)
    
    def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
        """Alias untuk membersihkan data preprocessed.
        
        Args:
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test').
                       Jika None, semua split akan dibersihkan.
                       
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        return self.cleanup_data(target='preprocessed', target_split=target_split)
    
    def _remove_augmented_files(self, aug_dir: Path) -> int:
        """Remove augmented files (.jpg dan .txt)"""
        count = 0
        
        # Remove .jpg dari images
        images_dir = aug_dir / 'images'
        if images_dir.exists():
            for file_path in images_dir.glob('aug_*.jpg'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        # Remove .txt dari labels
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
        """Remove preprocessed files (.npy dan .txt)"""
        count = 0
        
        # Remove .npy dari images (TANPA .jpg)
        images_dir = prep_dir / 'images'
        if images_dir.exists():
            for file_path in images_dir.glob('pre_*.npy'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        # Remove .txt dari labels
        labels_dir = prep_dir / 'labels'
        if labels_dir.exists():
            for file_path in labels_dir.glob('pre_*.txt'):
                try:
                    file_path.unlink()
                    count += 1
                except Exception:
                    continue
        
        return count