"""
File: smartcash/dataset/augmentor/utils/paths.py
Deskripsi: Fixed path resolution dengan Google Drive priority dan one-liner optimizations
"""

import os
from pathlib import Path
from typing import Dict, List

# One-liner path builders dengan Google Drive resolution
get_images_dir = lambda base: f"{_resolve_drive_path(base)}/images"
get_labels_dir = lambda base: f"{_resolve_drive_path(base)}/labels" 
get_split_dir = lambda base, split: f"{_resolve_drive_path(base)}/{split}"
join_path = lambda *parts: os.path.join(*parts)
ensure_path = lambda path: Path(_resolve_drive_path(path)).mkdir(parents=True, exist_ok=True) or str(_resolve_drive_path(path))

def _resolve_drive_path(path: str) -> str:
    """Resolve path dengan Google Drive priority."""
    if os.path.exists(path):
        return path
    
    # Check jika path sudah absolute dan valid
    if path.startswith('/'):
        return path
    
    # Try resolve ke Google Drive locations
    drive_bases = [
        '/content/drive/MyDrive/SmartCash',
        '/content/drive/MyDrive',
        '/content'
    ]
    
    for base in drive_bases:
        resolved = os.path.join(base, path) if not path.startswith(base) else path
        if os.path.exists(resolved):
            return resolved
        
        # Check parent directory exists (untuk creation)
        parent = os.path.dirname(resolved)
        if os.path.exists(parent):
            return resolved
    
    return path  # Fallback ke original

# One-liner path validators
path_exists = lambda path: Path(_resolve_drive_path(path)).exists()
is_directory = lambda path: Path(_resolve_drive_path(path)).is_dir()
is_file = lambda path: Path(_resolve_drive_path(path)).is_file() 
is_symlink = lambda path: Path(_resolve_drive_path(path)).is_symlink()

# One-liner path extractors
get_stem = lambda path: Path(path).stem
get_parent = lambda path: str(Path(path).parent)
get_suffix = lambda path: Path(path).suffix
get_basename = lambda path: Path(path).name

def resolve_augmentation_paths(raw_dir: str, aug_dir: str, prep_dir: str, split: str = "train") -> Dict[str, str]:
    """
    Resolve semua path dengan Google Drive priority dan one-liners.
    
    Args:
        raw_dir: Direktori raw data
        aug_dir: Direktori augmented output
        prep_dir: Direktori preprocessed target
        split: Split dataset
        
    Returns:
        Dictionary berisi semua resolved path
    """
    # Resolve all paths to Google Drive
    resolved_raw = _resolve_drive_path(raw_dir)
    resolved_aug = _resolve_drive_path(aug_dir)
    resolved_prep = _resolve_drive_path(prep_dir)
    
    return {
        # Input paths (dari raw data) - resolved
        'raw_dir': resolved_raw,
        'raw_images': get_images_dir(resolved_raw),
        'raw_labels': get_labels_dir(resolved_raw),
        
        # Output paths (ke augmented) - resolved
        'aug_dir': resolved_aug,
        'aug_images': get_images_dir(resolved_aug),
        'aug_labels': get_labels_dir(resolved_aug),
        
        # Final paths (ke preprocessed dengan split) - resolved
        'prep_dir': resolved_prep,
        'prep_split_dir': get_split_dir(resolved_prep, split),
        'prep_images': join_path(resolved_prep, split, 'images'),
        'prep_labels': join_path(resolved_prep, split, 'labels'),
        
        # Metadata
        'split': split,
        'drive_resolved': True
    }

def ensure_augmentation_directories(paths: Dict[str, str]) -> List[str]:
    """
    Pastikan semua direktori output ada dengan Google Drive support.
    
    Args:
        paths: Dictionary path hasil resolve_augmentation_paths
        
    Returns:
        List direktori yang dibuat
    """
    dirs_to_create = ['aug_images', 'aug_labels', 'prep_images', 'prep_labels']
    created_dirs = []
    
    for key in dirs_to_create:
        if key in paths:
            dir_path = paths[key]
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
            except Exception:
                pass  # Silent fail untuk permission issues
    
    return created_dirs

def find_raw_images(raw_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Cari semua file gambar di direktori raw dengan Google Drive resolution.
    
    Args:
        raw_dir: Direktori raw data
        extensions: List ekstensi yang didukung
        
    Returns:
        List path file gambar
    """
    if extensions is None: 
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    resolved_raw = _resolve_drive_path(raw_dir)
    
    if not os.path.exists(resolved_raw):
        return []
    
    try:
        raw_path = Path(resolved_raw)
        
        # Try standard structure first: raw_dir/images/
        images_dir = raw_path / 'images'
        if images_dir.exists():
            return [str(f) for ext in extensions for f in images_dir.glob(f'*{ext}') if f.is_file()]
        
        # Fallback to root directory
        return [str(f) for ext in extensions for f in raw_path.glob(f'*{ext}') if f.is_file()]
        
    except Exception:
        return []

def find_corresponding_label(image_path: str, labels_dir: str = None) -> str:
    """
    Cari file label dengan smart detection dan Google Drive support.
    
    Args:
        image_path: Path file gambar
        labels_dir: Optional direktori label
        
    Returns:
        Path file label atau empty string jika tidak ada
    """
    image_path = _resolve_drive_path(image_path)
    img_stem = get_stem(image_path)
    img_parent = Path(image_path).parent
    
    # Smart label directory detection
    potential_label_dirs = []
    
    if labels_dir:
        potential_label_dirs.append(_resolve_drive_path(labels_dir))
    
    # Add common patterns
    potential_label_dirs.extend([
        str(img_parent.parent / 'labels'),  # Standard YOLO
        str(img_parent.parent / 'label'),   # Alternative
        str(img_parent / 'labels'),         # Same level
        str(img_parent.parent),             # Root mixed
        str(img_parent)                     # Same directory
    ])
    
    # Find label file
    for label_dir in potential_label_dirs:
        label_path = os.path.join(label_dir, f"{img_stem}.txt")
        if os.path.exists(label_path):
            return label_path
    
    return ""

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
    Build path output dengan Google Drive resolution.
    
    Args:
        base_path: Base directory (aug_dir)
        filename: Nama file tanpa extension
        has_labels: Apakah ada file label
        
    Returns:
        Dictionary berisi resolved path output
    """
    resolved_base = _resolve_drive_path(base_path)
    
    paths = {
        'image': join_path(get_images_dir(resolved_base), f"{filename}.jpg")
    }
    
    if has_labels:
        paths['label'] = join_path(get_labels_dir(resolved_base), f"{filename}.txt")
    
    return paths

# One-liner path cleaners dengan Google Drive support
find_files_with_prefix = lambda directory, prefix: [str(f) for f in Path(_resolve_drive_path(directory)).rglob(f'{prefix}_*.*') if f.is_file()]
count_files_with_prefix = lambda directory, prefix: len(find_files_with_prefix(directory, prefix))
get_file_size = lambda path: Path(_resolve_drive_path(path)).stat().st_size if path_exists(path) else 0

def validate_augmentation_structure(paths: Dict[str, str]) -> Dict[str, bool]:
    """
    Validasi struktur direktori dengan Google Drive resolution.
    
    Args:
        paths: Dictionary path hasil resolve_augmentation_paths
        
    Returns:
        Dictionary hasil validasi
    """
    return {
        'raw_exists': path_exists(paths.get('raw_dir', '')),
        'raw_images_exists': path_exists(paths.get('raw_images', '')),
        'raw_has_files': len(find_raw_images(paths.get('raw_dir', ''))) > 0,
        'aug_dir_ready': is_directory(paths.get('aug_dir', '')) or path_exists(get_parent(paths.get('aug_dir', ''))),
        'prep_dir_ready': is_directory(paths.get('prep_dir', '')) or path_exists(get_parent(paths.get('prep_dir', ''))),
        'drive_mounted': '/content/drive/MyDrive' in paths.get('raw_dir', ''),
        'paths_resolved': paths.get('drive_resolved', False)
    }

def get_best_data_location() -> str:
    """
    Dapatkan lokasi data terbaik berdasarkan availability Google Drive.
    
    Returns:
        Path ke lokasi data terbaik
    """
    # Priority order: Drive mounted > Drive unmounted > Local Colab
    candidates = [
        '/content/drive/MyDrive/SmartCash/data',
        '/content/drive/MyDrive/data', 
        '/content/SmartCash/data',
        '/content/data',
        'data'
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        # Check if parent exists (untuk creation possibility)
        parent = os.path.dirname(candidate)
        if os.path.exists(parent):
            return candidate
    
    return 'data'  # Ultimate fallback

def get_google_drive_info() -> Dict[str, Any]:
    """
    Dapatkan informasi Google Drive mount status.
    
    Returns:
        Dictionary dengan Drive info
    """
    drive_path = '/content/drive/MyDrive'
    smartcash_path = '/content/drive/MyDrive/SmartCash'
    
    return {
        'drive_mounted': os.path.exists(drive_path),
        'smartcash_folder_exists': os.path.exists(smartcash_path),
        'recommended_data_path': get_best_data_location(),
        'mount_command': 'from google.colab import drive; drive.mount("/content/drive")',
        'setup_command': f'!mkdir -p "{smartcash_path}/data"'
    }

# Advanced path utilities dengan Google Drive optimization
resolve_relative_path = lambda base, relative: str(Path(_resolve_drive_path(base)) / relative)
get_common_parent = lambda *paths: str(Path(os.path.commonpath([_resolve_drive_path(str(p)) for p in paths])))
make_relative_to = lambda path, base: str(Path(_resolve_drive_path(path)).relative_to(Path(_resolve_drive_path(base))))

# One-liner Google Drive helpers
is_drive_path = lambda path: '/content/drive/MyDrive' in str(path)
to_drive_path = lambda path: _resolve_drive_path(path)
ensure_drive_structure = lambda: [Path(p).mkdir(parents=True, exist_ok=True) for p in ['/content/drive/MyDrive/SmartCash/data/images', '/content/drive/MyDrive/SmartCash/data/labels', '/content/drive/MyDrive/SmartCash/data/augmented/images', '/content/drive/MyDrive/SmartCash/data/augmented/labels']]