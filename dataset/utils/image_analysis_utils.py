"""
File: smartcash/dataset/utils/image_analysis_utils.py
Deskripsi: Utilitas untuk analisis gambar dan statistik dalam dataset
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

def count_files_by_prefix(dataset_dir: str, split_name: str = 'train', file_extension: str = '.jpg') -> Dict[str, int]:
    """
    Hitung jumlah file berdasarkan prefix.
    
    Args:
        dataset_dir: Path ke direktori dataset
        split_name: Nama split ('train', 'valid', 'test')
        file_extension: Ekstensi file yang dihitung
        
    Returns:
        Dictionary berisi {prefix: jumlah_file}
    """
    images_dir = os.path.join(dataset_dir, split_name, 'images')
    
    if not os.path.exists(images_dir):
        return {}
        
    # Hitung file berdasarkan prefix
    prefix_counts = {}
    
    for image_file in Path(images_dir).glob(f'*{file_extension}'):
        # Ambil prefix (karakter sebelum underscore pertama)
        parts = image_file.stem.split('_', 1)
        if parts:
            prefix = parts[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    
    return prefix_counts

def analyze_class_distribution(dataset_dir: str, split_name: str = 'train') -> Dict[str, int]:
    """
    Analisis distribusi kelas dalam dataset.
    
    Args:
        dataset_dir: Path ke direktori dataset
        split_name: Nama split ('train', 'valid', 'test')
        
    Returns:
        Dictionary berisi {class_id: jumlah_instance}
    """
    class_counts = {}
    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    
    if not os.path.exists(labels_dir):
        return class_counts
    
    # Loop melalui semua file label
    for label_file in Path(labels_dir).glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except Exception:
            # Skip file yang bermasalah
            continue
    
    return class_counts

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
    class_counts_all = {}
    class_counts_orig = {}
    class_counts_aug = {}
    
    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    images_dir = os.path.join(dataset_dir, split_name, 'images')
    
    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        return class_counts_all, class_counts_orig, class_counts_aug
    
    # Loop melalui semua file label
    for label_file in Path(labels_dir).glob('*.txt'):
        try:
            # Dapatkan nama file gambar yang bersesuaian
            img_name = label_file.stem + '.jpg'
            img_path = os.path.join(images_dir, img_name)
            
            # Tentukan apakah ini file original atau augmentasi
            is_augmented = label_file.stem.startswith(aug_prefix)
            is_original = label_file.stem.startswith(orig_prefix)
            
            # Baca file label
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        
                        # Update semua counter
                        class_counts_all[class_id] = class_counts_all.get(class_id, 0) + 1
                        
                        if is_augmented:
                            class_counts_aug[class_id] = class_counts_aug.get(class_id, 0) + 1
                        elif is_original:
                            class_counts_orig[class_id] = class_counts_orig.get(class_id, 0) + 1
        except Exception:
            # Skip file yang bermasalah
            continue
    
    return class_counts_all, class_counts_orig, class_counts_aug