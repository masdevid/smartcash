"""
File: smartcash/ui/dataset/augmentation/utils/validation_utils.py
Deskripsi: Utilitas validasi untuk augmentasi dataset
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

def validate_augmentation_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi augmentasi
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Validasi augmentasi diaktifkan
    if not config.get('enabled', True):
        return False, "Augmentasi tidak diaktifkan"
    
    # Validasi jenis augmentasi
    aug_types = config.get('types', [])
    if not aug_types or not isinstance(aug_types, (list, tuple)):
        return False, "Jenis augmentasi tidak valid"
    
    valid_types = ['combined', 'position', 'lighting']
    invalid_types = [t for t in aug_types if t not in valid_types]
    if invalid_types:
        return False, f"Jenis augmentasi tidak valid: {', '.join(invalid_types)}"
    
    # Validasi num_variations
    num_variations = config.get('num_variations', 0)
    if not isinstance(num_variations, int) or num_variations <= 0:
        return False, "Jumlah variasi harus berupa integer positif"
    
    if num_variations > 10:
        return False, "Jumlah variasi maksimal adalah 10"
    
    # Validasi target_count
    target_count = config.get('target_count', 0)
    if not isinstance(target_count, int) or target_count <= 0:
        return False, "Target count harus berupa integer positif"
    
    if target_count > 10000:
        return False, "Target count maksimal adalah 10000"
    
    # Validasi output_prefix
    output_prefix = config.get('output_prefix', '')
    if not output_prefix or not isinstance(output_prefix, str):
        return False, "Output prefix tidak boleh kosong"
    
    # Validasi karakter output_prefix
    if not output_prefix.replace('_', '').replace('-', '').isalnum():
        return False, "Output prefix hanya boleh berisi huruf, angka, underscore, dan dash"
    
    # Validasi output_dir
    output_dir = config.get('output_dir', '')
    if not output_dir:
        return False, "Output directory tidak boleh kosong"
    
    return True, ""

def validate_dataset_availability(ui_components: Dict[str, Any], split: str, data_dir: str = 'data') -> Tuple[bool, str]:
    """
    Validasi ketersediaan dataset untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset (train, valid, test)
        data_dir: Direktori data
        
    Returns:
        Tuple (is_available, error_message)
    """
    # Validasi split
    if split not in ['train', 'valid', 'test']:
        return False, f"Split '{split}' tidak valid. Harus salah satu dari: train, valid, test"
    
    # Path ke direktori split
    split_dir = os.path.join(data_dir, 'preprocessed', split)
    
    # Cek direktori split
    if not os.path.exists(split_dir):
        return False, f"Direktori dataset {split} tidak ditemukan: {split_dir}"
    
    # Cek direktori images
    images_dir = os.path.join(split_dir, 'images')
    if not os.path.exists(images_dir):
        return False, f"Direktori images tidak ditemukan: {images_dir}"
    
    # Cek direktori labels
    labels_dir = os.path.join(split_dir, 'labels')
    if not os.path.exists(labels_dir):
        return False, f"Direktori labels tidak ditemukan: {labels_dir}"
    
    # Hitung jumlah file
    try:
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        label_files = [f for f in os.listdir(labels_dir) 
                      if f.lower().endswith('.txt')]
        
        if not image_files:
            return False, f"Tidak ada file gambar di {images_dir}"
        
        if not label_files:
            return False, f"Tidak ada file label di {labels_dir}"
        
        # Cek rasio image:label
        if len(image_files) != len(label_files):
            log_message(ui_components, 
                       f"Jumlah gambar ({len(image_files)}) tidak sama dengan label ({len(label_files)})", 
                       "warning", "⚠️")
        
        return True, f"Dataset {split} siap: {len(image_files)} gambar, {len(label_files)} label"
        
    except Exception as e:
        return False, f"Error saat memeriksa dataset: {str(e)}"

def validate_disk_space(ui_components: Dict[str, Any], output_dir: str, estimated_size_mb: int = 1000) -> Tuple[bool, str]:
    """
    Validasi ruang disk yang tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori output
        estimated_size_mb: Estimasi ukuran output dalam MB
        
    Returns:
        Tuple (has_enough_space, message)
    """
    try:
        import shutil
        
        # Dapatkan ruang disk yang tersedia
        free_space_bytes = shutil.disk_usage(output_dir).free
        free_space_mb = free_space_bytes / (1024 * 1024)
        
        # Tambahkan buffer 20%
        required_space_mb = estimated_size_mb * 1.2
        
        if free_space_mb < required_space_mb:
            return False, f"Ruang disk tidak cukup. Tersedia: {free_space_mb:.0f}MB, Dibutuhkan: {required_space_mb:.0f}MB"
        
        return True, f"Ruang disk cukup: {free_space_mb:.0f}MB tersedia"
        
    except Exception as e:
        # Jika tidak bisa cek disk space, anggap OK
        log_message(ui_components, f"Tidak bisa cek ruang disk: {str(e)}", "warning", "⚠️")
        return True, "Status ruang disk tidak dapat dipastikan"

def validate_ui_components(ui_components: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validasi komponen UI yang diperlukan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (is_valid, missing_components)
    """
    required_components = [
        'progress_bar', 'progress_container', 'log_output', 
        'status_panel', 'confirmation_area'
    ]
    
    missing_components = []
    
    for component in required_components:
        if component not in ui_components:
            missing_components.append(component)
    
    return len(missing_components) == 0, missing_components

def validate_augmentation_parameters(ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi semua parameter augmentasi dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Validasi split
    split = getattr(ui_components.get('split_selector', {}), 'value', None)
    if not split:
        return False, "Split dataset tidak dipilih"
    
    # Validasi jenis augmentasi
    aug_types = getattr(ui_components.get('augmentation_types', {}), 'value', None)
    if not aug_types:
        return False, "Jenis augmentasi tidak dipilih"
    
    # Validasi num_variations
    num_variations = getattr(ui_components.get('num_variations', {}), 'value', 0)
    if num_variations <= 0:
        return False, "Jumlah variasi harus lebih dari 0"
    
    # Validasi target_count
    target_count = getattr(ui_components.get('target_count', {}), 'value', 0)
    if target_count <= 0:
        return False, "Target count harus lebih dari 0"
    
    # Validasi output_prefix
    output_prefix = getattr(ui_components.get('output_prefix', {}), 'value', '')
    if not output_prefix:
        return False, "Output prefix tidak boleh kosong"
    
    # Validasi dataset
    data_dir = ui_components.get('data_dir', 'data')
    is_available, error_msg = validate_dataset_availability(ui_components, split, data_dir)
    if not is_available:
        return False, error_msg
    
    return True, "Semua parameter valid"

def get_validation_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan ringkasan validasi lengkap.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary ringkasan validasi
    """
    summary = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Validasi parameter
    param_valid, param_error = validate_augmentation_parameters(ui_components)
    if not param_valid:
        summary['is_valid'] = False
        summary['errors'].append(f"Parameter: {param_error}")
    
    # Validasi komponen UI
    ui_valid, missing_components = validate_ui_components(ui_components)
    if not ui_valid:
        summary['warnings'].append(f"Komponen UI hilang: {', '.join(missing_components)}")
    
    # Validasi ruang disk
    output_dir = getattr(ui_components.get('output_dir', {}), 'value', 'data/augmented')
    disk_valid, disk_msg = validate_disk_space(ui_components, output_dir)
    if not disk_valid:
        summary['warnings'].append(f"Ruang disk: {disk_msg}")
    else:
        summary['info'].append(disk_msg)
    
    return summary