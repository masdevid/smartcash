"""
File: smartcash/dataset/augmentor/utils/path_operations.py
Deskripsi: SRP module untuk operasi path dengan smart resolution dan drive detection
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# PATH RESOLUTION - Smart drive path resolution
# =============================================================================

def resolve_drive_path(path: str) -> str:
    """Smart path resolution dengan prioritas Drive → Content → Local"""
    if os.path.isabs(path):
        return path
        
    # Priority search bases
    search_bases = [
        '/content/drive/MyDrive/SmartCash', 
        '/content/drive/MyDrive', 
        '/content/SmartCash', 
        '/content', 
        os.getcwd()
    ]
    
    for base in search_bases:
        full_path = os.path.join(base, path)
        if os.path.exists(full_path):
            return full_path
    
    return path

def find_dataset_directories(base_path: str) -> List[str]:
    """Find dataset directories dengan comprehensive search patterns"""
    resolved_path = resolve_drive_path(base_path)
    patterns = [
        resolved_path, 
        f"{resolved_path}/data", f"{resolved_path}/dataset", f"{resolved_path}/images"
    ]
    
    # Add split patterns
    for split in ['train', 'valid', 'test', 'val']:
        patterns.extend([
            f"{resolved_path}/{split}", f"{resolved_path}/{split}/images",
            f"{resolved_path}/data/{split}", f"{resolved_path}/data/{split}/images"
        ])
    
    return [p for p in set(patterns) if os.path.exists(p) and os.path.isdir(p)]

def smart_find_images(base_path: str, extensions: List[str] = None) -> List[str]:
    """Smart image finder dengan fallback strategies"""
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    # Strategy 1: Direct scan dari dataset directories
    for dir_path in find_dataset_directories(base_path):
        try:
            # Scan current + images subdirectory
            search_dirs = [dir_path, f"{dir_path}/images"] if os.path.exists(f"{dir_path}/images") else [dir_path]
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    image_files.extend([
                        os.path.join(search_dir, f) for f in os.listdir(search_dir)
                        if os.path.isfile(os.path.join(search_dir, f)) and Path(f).suffix.lower() in extensions
                    ])
        except (PermissionError, OSError):
            continue
    
    # Strategy 2: Recursive glob fallback
    if not image_files:
        try:
            resolved_path = resolve_drive_path(base_path)
            for ext in extensions:
                image_files.extend(glob.glob(f"{resolved_path}/**/*{ext}", recursive=True))
        except Exception:
            pass
    
    return list(set(image_files))

# =============================================================================
# PATH BUILDERS - One-liner utilities
# =============================================================================

build_paths = lambda raw_dir, aug_dir, prep_dir, split="train": {
    'raw_dir': resolve_drive_path(raw_dir), 'aug_dir': resolve_drive_path(aug_dir), 
    'prep_dir': resolve_drive_path(prep_dir), 'aug_images': f"{resolve_drive_path(aug_dir)}/images",
    'aug_labels': f"{resolve_drive_path(aug_dir)}/labels", 'prep_split': f"{resolve_drive_path(prep_dir)}/{split}"
}

build_split_aware_paths = lambda raw_dir, aug_dir, prep_dir, split="train": {
    **build_paths(raw_dir, aug_dir, prep_dir, split),
    'raw_split': f"{resolve_drive_path(raw_dir)}/{split}",
    'aug_split': f"{resolve_drive_path(aug_dir)}/{split}",
    'prep_split': f"{resolve_drive_path(prep_dir)}/{split}",
    'raw_images': f"{resolve_drive_path(raw_dir)}/{split}/images",
    'raw_labels': f"{resolve_drive_path(raw_dir)}/{split}/labels",
    'aug_images': f"{resolve_drive_path(aug_dir)}/{split}/images",
    'aug_labels': f"{resolve_drive_path(aug_dir)}/{split}/labels",
    'prep_images': f"{resolve_drive_path(prep_dir)}/{split}/images",
    'prep_labels': f"{resolve_drive_path(prep_dir)}/{split}/labels"
}

ensure_dirs = lambda *paths: [Path(resolve_drive_path(p)).mkdir(parents=True, exist_ok=True) for p in paths]
ensure_split_dirs = lambda base_dir, split: [Path(f"{resolve_drive_path(base_dir)}/{split}/{subdir}").mkdir(parents=True, exist_ok=True) for subdir in ['images', 'labels']]

# =============================================================================
# PATH VALIDATORS - One-liner checks
# =============================================================================

path_exists = lambda path: Path(resolve_drive_path(path)).exists()
get_stem = lambda path: Path(path).stem
get_parent = lambda path: str(Path(path).parent)
get_split_path = lambda base_dir, split: f"{resolve_drive_path(base_dir)}/{split}"
list_available_splits = lambda base_dir: [d.name for d in Path(resolve_drive_path(base_dir)).iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]

def get_best_data_location() -> str:
    """Smart data location dengan comprehensive search"""
    search_paths = [
        '/content/drive/MyDrive/SmartCash/data', '/content/drive/MyDrive/data', 
        '/content/SmartCash/data', '/content/data', 'data'
    ]
    for path in search_paths:
        if smart_find_images(path): 
            return path
        elif path_exists(path): 
            return path
    return 'data'