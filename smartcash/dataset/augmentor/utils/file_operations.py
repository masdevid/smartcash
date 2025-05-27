"""
File: smartcash/dataset/augmentor/utils/file_operations.py
Deskripsi: SRP module untuk operasi file dengan copy, move, dan validation
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Import resolve_drive_path dari path_operations yang sudah diperbaiki
from smartcash.dataset.augmentor.utils.path_operations import resolve_drive_path, path_exists, get_parent

# =============================================================================
# FILE OPERATIONS - One-liner utilities
# =============================================================================

find_images = lambda dir_path, exts=['.jpg', '.jpeg', '.png']: [str(f) for f in Path(resolve_drive_path(dir_path)).glob('**/*') if f.suffix.lower() in exts and f.is_file()]
find_labels = lambda dir_path: find_images(dir_path, ['.txt'])
find_aug_files = lambda dir_path: [str(f) for f in Path(resolve_drive_path(dir_path)).rglob('aug_*.*') if f.is_file()]

copy_file = lambda src, dst: shutil.copy2(resolve_drive_path(src), resolve_drive_path(dst)) if path_exists(src) else False
safe_copy_file = lambda src, dst: _safe_execute(lambda: copy_file(src, dst), False)
delete_file = lambda path: Path(resolve_drive_path(path)).unlink() if path_exists(path) else False
get_file_size = lambda path: Path(resolve_drive_path(path)).stat().st_size if path_exists(path) else 0

def copy_file_with_uuid_preservation(src: str, dst: str) -> bool:
    """Copy file dengan UUID preservation - reuse dari existing implementation"""
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve_drive_path(src), resolve_drive_path(dst))
        return True
    except Exception:
        return False

# =============================================================================
# IMAGE OPERATIONS - One-liner utilities
# =============================================================================

read_image = lambda path: cv2.imread(resolve_drive_path(path)) if path_exists(path) else None
save_image = lambda img, path: cv2.imwrite(resolve_drive_path(path), img) if img is not None else False
validate_image = lambda img: img is not None and img.size > 0
resize_image = lambda img, size: cv2.resize(img, size) if validate_image(img) else None

def safe_read_image(path: str) -> np.ndarray:
    """Safe image reading dengan fallback"""
    return _safe_execute(lambda: read_image(path), None)

# =============================================================================
# AUGMENTED FILE FINDERS - Split aware
# =============================================================================

def find_augmented_files_split_aware(aug_dir: str, target_split: str = None) -> List[str]:
    """Find augmented files dengan split awareness"""
    resolved_dir = resolve_drive_path(aug_dir)
    aug_files = []
    
    search_dirs = (
        [f"{resolved_dir}/{target_split}/images", f"{resolved_dir}/{target_split}"] if target_split
        else [f"{resolved_dir}/{split}/images" for split in ['train', 'valid', 'test'] if path_exists(f"{resolved_dir}/{split}/images")]
    )
    
    for search_dir in search_dirs:
        if path_exists(search_dir):
            aug_files.extend([str(f) for f in Path(search_dir).glob('aug_*.jpg')])
    
    return aug_files

def smart_find_images_split_aware(base_path: str, target_split: str = None, extensions: List[str] = None) -> List[str]:
    """Smart image finder dengan split awareness"""
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    resolved_path = resolve_drive_path(base_path)
    
    if target_split:
        # Search specific split
        search_paths = [
            f"{resolved_path}/{target_split}/images",
            f"{resolved_path}/{target_split}",
            f"{resolved_path}/images/{target_split}"
        ]
        
        for search_path in search_paths:
            if path_exists(search_path):
                return [
                    f"{search_path}/{f}" for f in os.listdir(search_path)
                    if Path(f).suffix.lower() in extensions
                ]
    
    # Fallback to comprehensive search
    from smartcash.dataset.augmentor.utils.path_operations import smart_find_images
    return smart_find_images(base_path, extensions)

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _safe_execute(operation, fallback_result=None):
    """Safe execution dengan fallback"""
    try:
        return operation()
    except Exception:
        return fallback_result