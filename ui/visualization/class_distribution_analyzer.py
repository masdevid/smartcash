"""
File: smartcash/ui/visualization/class_distribution_analyzer.py
Deskripsi: Utilitas untuk analisis distribusi kelas dalam dataset
"""

import os
from typing import Dict, Tuple
from pathlib import Path

# Import fungsi dari utils terkonsolidasi
from smartcash.dataset.utils.image_analysis_utils import analyze_class_distribution, analyze_class_distribution_by_prefix, count_files_by_prefix

def analyze_class_distribution(dataset_dir: str, split_name: str = 'train') -> Dict[str, int]:
    """
    Analisis distribusi kelas dalam dataset.
    
    Args:
        dataset_dir: Path ke direktori dataset
        split_name: Nama split ('train', 'valid', 'test')
        
    Returns:
        Dictionary berisi {class_id: jumlah_instance}
    """
    # Delegasi ke fungsi di modul terkonsolidasi
    return analyze_class_distribution(dataset_dir, split_name)

def analyze_class_distribution_by_prefix(
    dataset_dir: str, 
    split_name: str = 'train',
    aug_prefix: str = 'aug',
    orig_prefix: str = 'rp'
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Analisis distribusi kelas berdasarkan prefix file.
    
    Args:
        dataset_dir: Path ke direktori dataset
        split_name: Nama split ('train', 'valid', 'test')
        aug_prefix: Prefix untuk file augmentasi
        orig_prefix: Prefix untuk file original
        
    Returns:
        Tuple berisi (class_counts_all, class_counts_orig, class_counts_aug)
    """
    # Delegasi ke fungsi di modul terkonsolidasi
    return analyze_class_distribution_by_prefix(dataset_dir, split_name, aug_prefix, orig_prefix)

def count_files_by_prefix(dataset_dir: str, split_name: str = 'train', file_extension: str = '.jpg'):
    """
    Hitung jumlah file berdasarkan prefix.
    
    Args:
        dataset_dir: Path ke direktori dataset
        split_name: Nama split ('train', 'valid', 'test')
        file_extension: Ekstensi file yang dihitung
        
    Returns:
        Dictionary berisi {prefix: jumlah_file}
    """
    # Delegasi ke fungsi di modul terkonsolidasi
    return count_files_by_prefix(dataset_dir, split_name, file_extension)