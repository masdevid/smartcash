"""
File: smartcash/dataset/augmentor/utils/cleaner.py
Deskripsi: Utility untuk cleanup augmented data dengan optimized operations
"""

import os
import glob
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

class AugmentedDataCleaner:
    """Utility untuk cleanup data augmented dengan progress tracking."""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        self.logger = self.comm.logger if self.comm else None
        
    def cleanup_all_augmented_files(self, include_preprocessed: bool = True) -> Dict[str, Any]:
        """
        Cleanup semua file augmented dengan prefix aug_*.
        
        Args:
            include_preprocessed: Apakah termasuk cleanup di preprocessed
            
        Returns:
            Dictionary hasil cleanup
        """
        if self.logger: self.logger.info("🧹 Memulai cleanup augmented files")
        if self.comm: self.comm.progress("overall", 5, 100, "Inisialisasi cleanup")
        
        start_time = time.time()
        cleanup_stats = defaultdict(int)
        
        try:
            # Get directories untuk cleanup
            directories_to_clean = self._get_cleanup_directories(include_preprocessed)
            
            if not directories_to_clean:
                return self._create_result('empty', 'Tidak ada directory untuk cleanup', cleanup_stats, 0)
            
            # Count total files untuk progress calculation
            if self.comm: self.comm.progress("overall", 10, 100, "Menghitung file untuk cleanup")
            total_files = self._count_augmented_files(directories_to_clean)
            
            if total_files == 0:
                return self._create_result('empty', 'Tidak ada file augmented untuk dihapus', cleanup_stats, 0)
            
            if self.logger: self.logger.info(f"📊 Ditemukan {total_files} file augmented untuk cleanup")
            
            # Cleanup directories satu per satu
            processed_files = 0
            for i, (dir_name, dir_path) in enumerate(directories_to_clean.items()):
                start_progress = 20 + (i * 70 // len(directories_to_clean))
                end_progress = 20 + ((i + 1) * 70 // len(directories_to_clean))
                
                if self.comm: self.comm.progress("overall", start_progress, 100, f"Cleanup {dir_name}")
                
                dir_stats = self._cleanup_directory(dir_path, start_progress, end_progress)
                cleanup_stats[f'{dir_name}_deleted'] = dir_stats['deleted']
                cleanup_stats['total_deleted'] += dir_stats['deleted']
                processed_files += dir_stats['processed']
            
            # Final summary
            processing_time = time.time() - start_time
            
            if self.comm: self.comm.progress("overall", 100, 100, f"Cleanup selesai: {cleanup_stats['total_deleted']} file")
            if self.logger: self.logger.success(f"✅ Cleanup selesai: {cleanup_stats['total_deleted']} file dalam {processing_time:.1f}s")
            
            return self._create_result('success', f"Berhasil menghapus {cleanup_stats['total_deleted']} file", cleanup_stats, processing_time)
            
        except Exception as e:
            error_msg = f"Error cleanup: {str(e)}"
            if self.logger: self.logger.error(f"❌ {error_msg}")
            if self.comm: self.comm.log("error", error_msg)
            return self._create_result('error', error_msg, cleanup_stats, time.time() - start_time)
    
    def _get_cleanup_directories(self, include_preprocessed: bool) -> Dict[str, str]:
        """Get directories yang perlu cleanup."""
        directories = {}
        
        # Augmented directory
        aug_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
        if os.path.exists(aug_dir):
            directories['augmented'] = aug_dir
        
        # Preprocessed directories jika diminta
        if include_preprocessed:
            prep_dir = self.config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            if os.path.exists(prep_dir):
                for split in ['train', 'valid', 'test']:
                    split_dir = os.path.join(prep_dir, split)
                    if os.path.exists(split_dir):
                        directories[f'preprocessed_{split}'] = split_dir
        
        return directories
    
    def _count_augmented_files(self, directories: Dict[str, str]) -> int:
        """Count total augmented files di semua directories."""
        total_count = 0
        
        for dir_name, dir_path in directories.items():
            try:
                count = len(find_aug_files(dir_path))
                total_count += count
                if self.logger: self.logger.debug(f"📁 {dir_name}: {count} file augmented")
            except Exception:
                continue
                
        return total_count
    
    def _cleanup_directory(self, directory: str, start_progress: int, end_progress: int) -> Dict[str, int]:
        """Cleanup single directory dengan progress tracking."""
        stats = {'deleted': 0, 'processed': 0, 'errors': 0}
        
        try:
            # Find augmented files
            aug_files = find_aug_files(directory)
            
            if not aug_files:
                return stats
            
            # Delete files dengan progress updates
            for i, file_path in enumerate(aug_files):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        stats['deleted'] += 1
                    stats['processed'] += 1
                    
                    # Update progress setiap 10% dari directory ini
                    if i % max(1, len(aug_files) // 10) == 0:
                        file_progress = int((i / len(aug_files)) * (end_progress - start_progress))
                        current_progress = start_progress + file_progress
                        if self.comm: self.comm.progress("overall", current_progress, 100, f"Deleted: {stats['deleted']}/{len(aug_files)}")
                        
                except Exception:
                    stats['errors'] += 1
                    continue
            
            # Cleanup empty directories
            self._cleanup_empty_directories(directory)
            
        except Exception as e:
            if self.logger: self.logger.warning(f"⚠️ Error cleanup directory {directory}: {str(e)}")
            
        return stats
    
    def _cleanup_empty_directories(self, base_dir: str) -> None:
        """Cleanup empty directories setelah file deletion."""
        try:
            for root, dirs, files in os.walk(base_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # Directory kosong
                            os.rmdir(dir_path)
                            if self.logger: self.logger.debug(f"🗑️ Removed empty directory: {dir_path}")
                    except Exception:
                        continue
        except Exception:
            pass
    
    def _create_result(self, status: str, message: str, stats: Dict, processing_time: float) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            'status': status,
            'message': message,
            'total_deleted': stats.get('total_deleted', 0),
            'processing_time': processing_time,
            'details': dict(stats),
            'timestamp': time.time()
        }
    
    def get_cleanup_preview(self, include_preprocessed: bool = True) -> Dict[str, Any]:
        """
        Preview file yang akan dihapus tanpa actual deletion.
        
        Args:
            include_preprocessed: Include preprocessed files
            
        Returns:
            Preview information
        """
        directories = self._get_cleanup_directories(include_preprocessed)
        preview = {'directories': {}, 'total_files': 0}
        
        for dir_name, dir_path in directories.items():
            try:
                files = find_aug_files(dir_path)
                preview['directories'][dir_name] = {
                    'path': dir_path,
                    'file_count': len(files),
                    'sample_files': files[:5]  # Sample 5 files
                }
                preview['total_files'] += len(files)
            except Exception:
                preview['directories'][dir_name] = {
                    'path': dir_path,
                    'file_count': 0,
                    'error': 'Cannot access directory'
                }
        
        return preview

# One-liner utilities
find_aug_files = lambda directory: glob.glob(os.path.join(directory, "**", "aug_*.*"), recursive=True)
count_aug_files = lambda directory: len(find_aug_files(directory))
delete_aug_files = lambda directory: [os.remove(f) for f in find_aug_files(directory) if os.path.exists(f)]

# Factory function
def create_augmented_cleaner(config: Dict[str, Any], communicator=None) -> AugmentedDataCleaner:
    """Factory function untuk create augmented data cleaner."""
    return AugmentedDataCleaner(config, communicator)