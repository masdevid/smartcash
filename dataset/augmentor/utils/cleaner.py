"""
File: smartcash/dataset/augmentor/utils/cleaner.py
Deskripsi: Cleaner utility untuk augmented data dengan one-liner operations dan progress tracking
"""

import os
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

from smartcash.common.logger import get_logger

class AugmentedDataCleaner:
    """Cleaner untuk augmented data dengan progress tracking yang optimized."""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        # One-liner logger setup dengan fallback
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
        self.stats = defaultdict(int)
    
    def cleanup_all_augmented_files(self, include_preprocessed: bool = True) -> Dict[str, Any]:
        """
        Cleanup semua augmented files dengan prefix aug_*.
        
        Args:
            include_preprocessed: Apakah cleanup juga preprocessed files
            
        Returns:
            Dictionary hasil cleanup
        """
        self.logger.info("🧹 Memulai cleanup augmented data")
        # One-liner progress start
        self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 5, 100, "Memulai cleanup augmented data")
        
        try:
            cleanup_results = {'total_deleted': 0, 'folders_cleaned': [], 'errors': []}
            
            # Get directories untuk cleanup
            aug_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
            prep_dir = self.config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            
            # Cleanup augmented directory
            if os.path.exists(aug_dir):
                aug_result = self._cleanup_directory(aug_dir, "augmented")
                cleanup_results['total_deleted'] += aug_result['deleted']
                cleanup_results['folders_cleaned'].extend(aug_result['folders'])
                cleanup_results['errors'].extend(aug_result['errors'])
                
                # One-liner progress update
                self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 50, 100, f"Cleanup augmented: {aug_result['deleted']} files")
            
            # Cleanup preprocessed directory jika diminta
            if include_preprocessed and os.path.exists(prep_dir):
                prep_result = self._cleanup_preprocessed_directory(prep_dir)
                cleanup_results['total_deleted'] += prep_result['deleted']
                cleanup_results['folders_cleaned'].extend(prep_result['folders'])
                cleanup_results['errors'].extend(prep_result['errors'])
                
                # One-liner progress update
                self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 90, 100, f"Cleanup preprocessed: {prep_result['deleted']} files")
            
            # Final result
            if cleanup_results['total_deleted'] > 0:
                success_msg = f"✅ Cleanup berhasil: {cleanup_results['total_deleted']} file dihapus"
                self.logger.success if hasattr(self.logger, 'success') else self.logger.info(success_msg)
                cleanup_results['status'] = 'success'
                cleanup_results['message'] = success_msg
            else:
                info_msg = "ℹ️ Tidak ada file augmented yang perlu dihapus"
                self.logger.info(info_msg)
                cleanup_results['status'] = 'success'
                cleanup_results['message'] = info_msg
            
            # One-liner final progress
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 100, 100, "Cleanup selesai")
            
            return cleanup_results
            
        except Exception as e:
            error_msg = f"Error pada cleanup: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            # One-liner error logging
            self.comm and hasattr(self.comm, 'log') and self.comm.log("error", error_msg)
            return {'status': 'error', 'message': error_msg, 'total_deleted': 0}
    
    def _cleanup_directory(self, directory: str, dir_type: str) -> Dict[str, Any]:
        """Cleanup directory dengan one-liner file operations."""
        self.logger.info(f"🗑️ Cleaning {dir_type} directory: {directory}")
        
        deleted_count = 0
        folders_cleaned = []
        errors = []
        
        try:
            # One-liner find all aug_ files
            aug_files = glob.glob(os.path.join(directory, '**', 'aug_*.*'), recursive=True)
            
            if not aug_files:
                return {'deleted': 0, 'folders': [], 'errors': []}
            
            # One-liner batch delete dengan error handling
            for file_path in aug_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"Error deleting {file_path}: {str(e)}")
            
            # Track cleaned folders
            cleaned_folders = set(os.path.dirname(f) for f in aug_files)
            folders_cleaned = [f"{dir_type}: {len(aug_files)} files" for _ in [None] if aug_files]  # One item list
            
            self.logger.info(f"🗑️ {dir_type} cleanup: {deleted_count} files deleted")
            
            return {'deleted': deleted_count, 'folders': folders_cleaned, 'errors': errors}
            
        except Exception as e:
            error_msg = f"Error cleaning {dir_type} directory: {str(e)}"
            self.logger.error(error_msg)
            return {'deleted': 0, 'folders': [], 'errors': [error_msg]}
    
    def _cleanup_preprocessed_directory(self, prep_dir: str) -> Dict[str, Any]:
        """Cleanup preprocessed directory dengan split handling."""
        self.logger.info(f"🗑️ Cleaning preprocessed directory: {prep_dir}")
        
        total_deleted = 0
        folders_cleaned = []
        errors = []
        
        # One-liner check for splits
        splits = ['train', 'valid', 'test']
        existing_splits = [split for split in splits if os.path.exists(os.path.join(prep_dir, split))]
        
        for split in existing_splits:
            split_dir = os.path.join(prep_dir, split)
            split_result = self._cleanup_directory(split_dir, f"preprocessed/{split}")
            
            total_deleted += split_result['deleted']
            folders_cleaned.extend(split_result['folders'])
            errors.extend(split_result['errors'])
        
        return {'deleted': total_deleted, 'folders': folders_cleaned, 'errors': errors}
    
    def get_cleanup_preview(self, include_preprocessed: bool = True) -> Dict[str, Any]:
        """Preview files yang akan dihapus tanpa menghapus."""
        self.logger.info("👁️ Preview cleanup augmented data")
        
        try:
            preview_results = {'total_files': 0, 'directories': {}, 'file_types': defaultdict(int)}
            
            # Get directories
            aug_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')  
            prep_dir = self.config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            
            # Preview augmented directory
            if os.path.exists(aug_dir):
                aug_preview = self._preview_directory_cleanup(aug_dir, "augmented")
                preview_results['directories']['augmented'] = aug_preview
                preview_results['total_files'] += aug_preview['file_count']
                
                # Update file types
                for ext, count in aug_preview['file_types'].items():
                    preview_results['file_types'][ext] += count
            
            # Preview preprocessed directory
            if include_preprocessed and os.path.exists(prep_dir):
                prep_preview = self._preview_preprocessed_cleanup(prep_dir)
                preview_results['directories']['preprocessed'] = prep_preview
                preview_results['total_files'] += prep_preview['total_files']
                
                # Update file types
                for ext, count in prep_preview['file_types'].items():
                    preview_results['file_types'][ext] += count
            
            preview_results['status'] = 'success'
            return preview_results
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'total_files': 0}
    
    def _preview_directory_cleanup(self, directory: str, dir_type: str) -> Dict[str, Any]:
        """Preview directory cleanup dengan one-liner file analysis."""
        aug_files = glob.glob(os.path.join(directory, '**', 'aug_*.*'), recursive=True)
        
        # One-liner file type analysis
        file_types = defaultdict(int)
        [file_types.update({Path(f).suffix.lower(): 1}) for f in aug_files]
        
        return {
            'directory_type': dir_type,
            'file_count': len(aug_files),
            'file_types': dict(file_types),
            'sample_files': aug_files[:5]  # First 5 files sebagai sample
        }
    
    def _preview_preprocessed_cleanup(self, prep_dir: str) -> Dict[str, Any]:
        """Preview preprocessed cleanup dengan split analysis."""
        splits = ['train', 'valid', 'test']
        existing_splits = [split for split in splits if os.path.exists(os.path.join(prep_dir, split))]
        
        total_files = 0
        file_types = defaultdict(int)
        split_details = {}
        
        for split in existing_splits:
            split_dir = os.path.join(prep_dir, split)
            split_preview = self._preview_directory_cleanup(split_dir, f"preprocessed/{split}")
            
            split_details[split] = split_preview
            total_files += split_preview['file_count']
            
            # Merge file types
            for ext, count in split_preview['file_types'].items():
                file_types[ext] += count
        
        return {
            'total_files': total_files,
            'file_types': dict(file_types),
            'splits': split_details
        }

# Factory function
def create_augmented_data_cleaner(config: Dict[str, Any], communicator=None) -> AugmentedDataCleaner:
    """Factory function untuk create cleaner."""
    return AugmentedDataCleaner(config, communicator)

# One-liner utility functions
cleanup_all_aug_files = lambda config, include_prep=True, comm=None: AugmentedDataCleaner(config, comm).cleanup_all_augmented_files(include_prep)
preview_cleanup = lambda config, include_prep=True, comm=None: AugmentedDataCleaner(config, comm).get_cleanup_preview(include_prep)
find_aug_files = lambda directory: glob.glob(os.path.join(directory, '**', 'aug_*.*'), recursive=True)
count_aug_files = lambda directory: len(find_aug_files(directory))