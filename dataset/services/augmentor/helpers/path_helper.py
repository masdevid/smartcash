"""
File: smartcash/dataset/services/augmentor/helpers/path_helper.py
Deskripsi: Helper untuk setup path direktori augmentasi dan validasi input
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path

def setup_paths(config: Dict[str, Any], split: str) -> Dict[str, str]:
    """
    Setup paths untuk input, output, dan lokasi lain yang dibutuhkan augmentasi.
    
    Args:
        config: Konfigurasi aplikasi
        split: Nama split dataset ('train', 'valid', 'test')
        
    Returns:
        Dictionary berisi path yang dibutuhkan
    """
    # Base paths dengan one-liner
    paths = {
        'preprocessed_dir': config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed'),
        'augmented_dir': config.get('augmentation', {}).get('output_dir', 'data/augmented')
    }
    
    # Pastikan direktori augmented tidak sama dengan preprocessed untuk menghindari konflik
    if paths['augmented_dir'] == paths['preprocessed_dir']:
        paths['augmented_dir'] = os.path.join(paths['preprocessed_dir'], 'augmented')
    
    # Derived paths dengan one-liner
    return {**paths, **{
        'input_dir': os.path.join(paths['preprocessed_dir'], split),
        'images_input_dir': os.path.join(paths['preprocessed_dir'], split, 'images'),
        'labels_input_dir': os.path.join(paths['preprocessed_dir'], split, 'labels'),
        'output_dir': paths['augmented_dir'],
        'images_output_dir': os.path.join(paths['augmented_dir'], split, 'images'),  # Tambahkan split ke path
        'labels_output_dir': os.path.join(paths['augmented_dir'], split, 'labels'),  # Tambahkan split ke path
        'split': split,
        'final_output_dir': paths['preprocessed_dir']
    }}

def ensure_output_directories(paths: Dict[str, str]) -> bool:
    """
    Pastikan direktori output ada dan siap digunakan.
    
    Args:
        paths: Dictionary path hasil setup_paths
        
    Returns:
        Boolean keberhasilan
    """
    try:
        # Buat direktori output yang diperlukan dengan one-liner
        [os.makedirs(paths[key], exist_ok=True) for key in ['images_output_dir', 'labels_output_dir']]
        return True
    except Exception:
        return False

def get_path_targets(paths: Dict[str, str], split: str, is_temp: bool = False) -> Dict[str, str]:
    """
    Dapatkan lokasi target untuk file augmentasi.
    
    Args:
        paths: Dictionary path hasil setup_paths
        split: Nama split dataset
        is_temp: Gunakan direktori temporary
        
    Returns:
        Dictionary lokasi target
    """
    # Gunakan preprocessed atau temporary
    base_dir = paths['augmented_dir'] if is_temp else paths['preprocessed_dir']
    
    return {
        'images_dir': os.path.join(base_dir, split, 'images') if not is_temp else os.path.join(base_dir, 'images'),
        'labels_dir': os.path.join(base_dir, split, 'labels') if not is_temp else os.path.join(base_dir, 'labels')
    }

def create_output_basenames(output_prefix: str, file_id: str, num_variations: int) -> List[str]:
    """
    Buat nama base untuk file output berdasarkan variasi.
    
    Args:
        output_prefix: Prefix untuk file output
        file_id: ID file (biasanya stem dari nama file)
        num_variations: Jumlah variasi yang dibuat
        
    Returns:
        List nama base untuk tiap variasi
    """
    return [f"{output_prefix}_{file_id}_var{var_idx+1}" for var_idx in range(num_variations)]