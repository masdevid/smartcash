"""
File: smartcash/dataset/utils/file_wrapper.py
Deskripsi: Wrapper untuk fungsi file utils dari common untuk memastikan backward compatibility dan DRY
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from smartcash.common.file_utils import get_file_utils

# Fungsi-fungsi wrapper untuk backward compatibility

def find_image_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Wrapper untuk FileUtils.find_image_files"""
    return get_file_utils().find_image_files(directory, recursive)

def find_matching_label(image_path: Union[str, Path], labels_dir: Union[str, Path]) -> Optional[Path]:
    """Wrapper untuk FileUtils.find_matching_label"""
    return get_file_utils().find_matching_label(image_path, labels_dir)

def copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> bool:
    """Wrapper untuk FileUtils.copy_file"""
    return get_file_utils().copy_file(src, dst, overwrite)

def copy_files(
    source_dir: Union[str, Path], 
    target_dir: Union[str, Path],
    file_list: Optional[List[Union[str, Path]]] = None,
    patterns: Optional[List[str]] = None,
    flatten: bool = False,
    show_progress: bool = True
) -> Dict[str, int]:
    """Wrapper untuk FileUtils.copy_files"""
    return get_file_utils().copy_files(source_dir, target_dir, file_list, patterns, flatten, show_progress)

def move_files(
    source_dir: Union[str, Path], 
    target_dir: Union[str, Path],
    file_list: Optional[List[Union[str, Path]]] = None,
    patterns: Optional[List[str]] = None,
    flatten: bool = False,
    show_progress: bool = True,
    overwrite: bool = False
) -> Dict[str, int]:
    """Wrapper untuk FileUtils.move_files"""
    return get_file_utils().move_files(source_dir, target_dir, file_list, patterns, flatten, show_progress, overwrite)

def backup_directory(source_dir: Union[str, Path], suffix: Optional[str] = None) -> Optional[Path]:
    """Wrapper untuk FileUtils.backup_directory"""
    return get_file_utils().backup_directory(source_dir, suffix)

def extract_zip(
    zip_path: Union[str, Path], 
    output_dir: Union[str, Path],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    remove_zip: bool = False,
    show_progress: bool = True
) -> Dict[str, int]:
    """Wrapper untuk FileUtils.extract_zip"""
    return get_file_utils().extract_zip(zip_path, output_dir, include_patterns, exclude_patterns, remove_zip, show_progress)

def find_corrupted_images(
    directory: Union[str, Path],
    recursive: bool = True,
    show_progress: bool = True
) -> List[Path]:
    """Wrapper untuk FileUtils.find_corrupted_images"""
    return get_file_utils().find_corrupted_images(directory, recursive, show_progress)

def ensure_dir(path: Union[str, Path]) -> Path:
    """Wrapper untuk FileUtils.ensure_dir"""
    return get_file_utils().ensure_dir(path)

def file_exists(path: Union[str, Path]) -> bool:
    """Wrapper untuk FileUtils.file_exists"""
    return get_file_utils().file_exists(path)

def file_size(path: Union[str, Path]) -> int:
    """Wrapper untuk FileUtils.file_size"""
    return get_file_utils().file_size(path)

def format_size(size_bytes: int) -> str:
    """Wrapper untuk FileUtils.format_size"""
    return get_file_utils().format_size(size_bytes)