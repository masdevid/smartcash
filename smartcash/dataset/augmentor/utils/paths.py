"""
File: smartcash/dataset/augmentor/utils/paths.py
Deskripsi: Path resolution one-liners untuk augmentasi dengan clean flow
"""

import os
from pathlib import Path
from typing import Dict, List

# One-liner path builders
get_images_dir = lambda base: f"{base}/images"
get_labels_dir = lambda base: f"{base}/labels" 
get_split_dir = lambda base, split: f"{base}/{split}"
join_path = lambda *parts: os.path.join(*parts)
ensure_path = lambda path: Path(path).mkdir(parents=True, exist_ok=True) or str(path)

# One-liner path validators
path_exists = lambda path: Path(path).exists()
is_directory = lambda path: Path(path).is_dir()
is_file = lambda path: Path(path).is_file()
is_symlink = lambda path: Path(path).is_symlink()

# One-liner path extractors
get_stem = lambda path: Path(path).stem
get_parent = lambda path: str(Path(path).parent)
get_suffix = lambda path: Path(path).suffix
get_basename = lambda path: Path(path).name

def resolve_augmentation_paths(raw_dir: str, aug_dir: str, prep_dir: str, split: str = "train") -> Dict[str, str]:
    """
    Resolve semua path yang dibutuhkan untuk augmentasi dengan one-liners.
    
    Args:
        raw_dir: Direktori raw data
        aug_dir: Direktori augmented output
        prep_dir: Direktori preprocessed target
        split: Split dataset
        
    Returns:
        Dictionary berisi semua path yang dibutuhkan
    """
    return {
        # Input paths (dari raw data)
        'raw_dir': raw_dir,
        'raw_images': get_images_dir(raw_dir),
        'raw_labels': get_labels_dir(raw_dir),
        
        # Output paths (ke augmented)
        'aug_dir': aug_dir,
        'aug_images': get_images_dir(aug_dir),
        'aug_labels': get_labels_dir(aug_dir),
        
        # Final paths (ke preprocessed dengan split)
        'prep_dir': prep_dir,
        'prep_split_dir': get_split_dir(prep_dir, split),
        'prep_images': join_path(prep_dir, split, 'images'),
        'prep_labels': join_path(prep_dir, split, 'labels'),
        
        # Metadata
        'split': split
    }

def ensure_augmentation_directories(paths: Dict[str, str]) -> List[str]:
    """
    Pastikan semua direktori output ada dengan one-liners.
    
    Args:
        paths: Dictionary path hasil resolve_augmentation_paths
        
    Returns:
        List direktori yang dibuat
    """
    dirs_to_create = ['aug_images', 'aug_labels', 'prep_images', 'prep_labels']
    return [ensure_path(paths[key]) for key in dirs_to_create if key in paths]

def find_raw_images(raw_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Cari semua file gambar di direktori raw dengan one-liners.
    
    Args:
        raw_dir: Direktori raw data
        extensions: List ekstensi yang didukung
        
    Returns:
        List path file gambar
    """
    if extensions is None: extensions = ['.jpg', '.jpeg', '.png']
    raw_path = Path(raw_dir)
    
    # One-liner untuk cari semua gambar
    return [str(f) for ext in extensions for f in raw_path.rglob(f'*{ext}')]

def find_corresponding_label(image_path: str, labels_dir: str) -> str:
    """
    Cari file label yang sesuai dengan gambar dengan one-liner.
    
    Args:
        image_path: Path file gambar
        labels_dir: Direktori label
        
    Returns:
        Path file label atau empty string jika tidak ada
    """
    label_name = f"{get_stem(image_path)}.txt"
    label_path = join_path(labels_dir, label_name)
    return label_path if path_exists(label_path) else ""

def get_augmented_filename(original_path: str, prefix: str, variation: int) -> str:
    """
    Generate nama file augmented dengan pattern konsisten.
    
    Args:
        original_path: Path file original
        prefix: Prefix augmentasi (e.g., 'aug')
        variation: Nomor variasi
        
    Returns:
        Nama file augmented tanpa extension
    """
    stem = get_stem(original_path)
    return f"{prefix}_{stem}_var{variation}"

def build_output_paths(base_path: str, filename: str, has_labels: bool = True) -> Dict[str, str]:
    """
    Build path output untuk image dan label dengan one-liners.
    
    Args:
        base_path: Base directory (aug_dir)
        filename: Nama file tanpa extension
        has_labels: Apakah ada file label
        
    Returns:
        Dictionary berisi path output
    """
    paths = {
        'image': join_path(get_images_dir(base_path), f"{filename}.jpg")
    }
    
    if has_labels:
        paths['label'] = join_path(get_labels_dir(base_path), f"{filename}.txt")
    
    return paths

# One-liner path cleaners untuk cleanup operations
find_files_with_prefix = lambda directory, prefix: [str(f) for f in Path(directory).rglob(f'{prefix}_*.*')]
count_files_with_prefix = lambda directory, prefix: len(find_files_with_prefix(directory, prefix))
get_file_size = lambda path: Path(path).stat().st_size if path_exists(path) else 0

def validate_augmentation_structure(paths: Dict[str, str]) -> Dict[str, bool]:
    """
    Validasi struktur direktori augmentasi dengan one-liners.
    
    Args:
        paths: Dictionary path hasil resolve_augmentation_paths
        
    Returns:
        Dictionary hasil validasi
    """
    return {
        'raw_exists': path_exists(paths.get('raw_dir', '')),
        'raw_images_exists': path_exists(paths.get('raw_images', '')),
        'aug_dir_ready': is_directory(paths.get('aug_dir', '')) or path_exists(get_parent(paths.get('aug_dir', ''))),
        'prep_dir_ready': is_directory(paths.get('prep_dir', '')) or path_exists(get_parent(paths.get('prep_dir', '')))
    }

# Advanced path utilities dengan one-liners
resolve_relative_path = lambda base, relative: str(Path(base) / relative)
get_common_parent = lambda *paths: str(Path(os.path.commonpath([str(p) for p in paths])))
make_relative_to = lambda path, base: str(Path(path).relative_to(Path(base)))