"""
File: smartcash/dataset/augmentor/utils/cleanup_operations.py
Deskripsi: SRP module untuk cleanup operations dengan split-aware dan UUID tracking
"""

from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.augmentor.utils.file_operations import find_aug_files, delete_file
from smartcash.dataset.augmentor.utils.progress_tracker import ProgressTracker

def cleanup_files(aug_dir: str, prep_dir: str = None, progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
    """Fixed cleanup dengan real-time progress updates"""
    total_deleted = 0
    errors = []
    
    aug_files = find_aug_files(aug_dir)
    if prep_dir:
        aug_files.extend(find_aug_files(prep_dir))
    
    total_files = len(aug_files)
    if progress_tracker:
        progress_tracker.progress("overall", 0, 100, f"Mulai cleanup: {total_files} file")
    
    for i, file_path in enumerate(aug_files):
        if delete_file(file_path):
            total_deleted += 1
        else:
            errors.append(f"Failed to delete {file_path}")
        
        # Real-time progress
        if progress_tracker:
            current_progress = int((i / max(total_files, 1)) * 100)
            progress_tracker.progress("overall", current_progress, 100, f"Cleanup: {i+1}/{total_files} file")
    
    if progress_tracker:
        progress_tracker.progress("overall", 100, 100, f"Cleanup selesai: {total_deleted} file dihapus")
    
    return {
        'status': 'success' if total_deleted > 0 else 'empty',
        'total_deleted': total_deleted,
        'message': f"Berhasil menghapus {total_deleted} file" if total_deleted > 0 else "Tidak ada file untuk dihapus",
        'errors': errors
    }

def cleanup_split_aware(aug_dir: str, prep_dir: str = None, target_split: str = None, 
                       progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
    """Cleanup dengan split awareness"""
    total_deleted = 0
    errors = []
    
    # Cleanup augmented files
    if target_split:
        aug_split_dir = f"{aug_dir}/{target_split}"
        deleted = _cleanup_split_directory(aug_split_dir)
        total_deleted += deleted
        if progress_tracker:
            progress_tracker.progress("overall", 50, 100, f"Cleanup {target_split}: {deleted} files")
    else:
        for split in ['train', 'valid', 'test']:
            aug_split_dir = f"{aug_dir}/{split}"
            if Path(aug_split_dir).exists():
                deleted = _cleanup_split_directory(aug_split_dir)
                total_deleted += deleted
    
    # Cleanup preprocessed augmented files
    if prep_dir:
        if target_split:
            prep_split_dir = f"{prep_dir}/{target_split}"
            deleted = _cleanup_augmented_from_split(prep_split_dir)
            total_deleted += deleted
        else:
            for split in ['train', 'valid', 'test']:
                prep_split_dir = f"{prep_dir}/{split}"
                if Path(prep_split_dir).exists():
                    deleted = _cleanup_augmented_from_split(prep_split_dir)
                    total_deleted += deleted
    
    if progress_tracker:
        progress_tracker.progress("overall", 100, 100, f"Split cleanup selesai: {total_deleted} files")
    
    return {
        'status': 'success' if total_deleted > 0 else 'empty',
        'total_deleted': total_deleted,
        'message': f"Split-aware cleanup: {total_deleted} file dihapus",
        'split_aware': True, 'target_split': target_split, 'errors': errors
    }

def _cleanup_split_directory(split_dir: str) -> int:
    """Cleanup single split directory"""
    deleted = 0
    split_path = Path(split_dir)
    
    if not split_path.exists():
        return deleted
    
    for subdir in ['images', 'labels']:
        subdir_path = split_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.glob('aug_*.*'):
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception:
                    pass
    
    return deleted

def _cleanup_augmented_from_split(split_dir: str) -> int:
    """Cleanup hanya augmented files dari split directory"""
    deleted = 0
    split_path = Path(split_dir)
    
    if not split_path.exists():
        return deleted
    
    for subdir in ['images', 'labels']:
        subdir_path = split_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.glob('aug_*.*'):
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception:
                    pass
    
    return deleted

# One-liner utilities
cleanup_augmented_files = lambda aug_dir, prep_dir=None: cleanup_files(aug_dir, prep_dir)
cleanup_split_data = lambda aug_dir, prep_dir, split: cleanup_split_aware(aug_dir, prep_dir, split)