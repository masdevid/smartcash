"""
File: smartcash/ui/dataset/augmentation/handlers/parameter_handler.py
Deskripsi: Handler parameter untuk validasi dan ekstraksi parameter augmentasi dataset
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from smartcash.common.logger import get_logger

def validate_augmentation_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi parameter augmentasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil validasi dengan status dan pesan
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    # Dapatkan konfigurasi dari UI
    from smartcash.ui.dataset.augmentation.handlers.config_handler import get_config_from_ui
    config = get_config_from_ui(ui_components)
    
    # Dapatkan konfigurasi augmentasi
    aug_config = config.get('augmentation', {})
    
    # Validasi parameter dasar
    if not aug_config.get('enabled', True):
        return {
            'status': 'error',
            'message': 'Augmentasi tidak diaktifkan'
        }
    
    # Validasi jenis augmentasi
    aug_types = aug_config.get('types', [])
    if not aug_types or not isinstance(aug_types, (list, tuple)) or len(aug_types) == 0:
        return {
            'status': 'error',
            'message': 'Jenis augmentasi tidak valid atau tidak dipilih'
        }
    
    # Validasi jumlah variasi
    num_variations = aug_config.get('num_variations', 0)
    if num_variations <= 0:
        return {
            'status': 'error',
            'message': 'Jumlah variasi harus lebih dari 0'
        }
    
    # Validasi target count
    target_count = aug_config.get('target_count', 0)
    if target_count <= 0:
        return {
            'status': 'error',
            'message': 'Target jumlah per kelas harus lebih dari 0'
        }
    
    # Validasi output prefix
    output_prefix = aug_config.get('output_prefix', '')
    if not output_prefix:
        return {
            'status': 'error',
            'message': 'Output prefix tidak boleh kosong'
        }
    
    # Validasi parameter posisi
    position_params = aug_config.get('position', {})
    
    # Validasi flip
    fliplr = position_params.get('fliplr', -1)
    if fliplr < 0 or fliplr > 1:
        return {
            'status': 'error',
            'message': 'Probabilitas flip horizontal harus antara 0 dan 1'
        }
    
    # Validasi rotasi
    degrees = position_params.get('degrees', -1)
    if degrees < 0:
        return {
            'status': 'error',
            'message': 'Derajat rotasi harus lebih dari atau sama dengan 0'
        }
    
    # Validasi translasi
    translate = position_params.get('translate', -1)
    if translate < 0 or translate > 1:
        return {
            'status': 'error',
            'message': 'Translasi harus antara 0 dan 1'
        }
    
    # Validasi skala
    scale = position_params.get('scale', -1)
    if scale < 0 or scale > 1:
        return {
            'status': 'error',
            'message': 'Skala harus antara 0 dan 1'
        }
    
    # Validasi shear
    shear_max = position_params.get('shear_max', -1)
    if shear_max < 0:
        return {
            'status': 'error',
            'message': 'Shear maksimum harus lebih dari atau sama dengan 0'
        }
    
    # Validasi parameter pencahayaan
    lighting_params = aug_config.get('lighting', {})
    
    # Validasi HSV
    hsv_h = lighting_params.get('hsv_h', -1)
    if hsv_h < 0 or hsv_h > 1:
        return {
            'status': 'error',
            'message': 'HSV Hue harus antara 0 dan 1'
        }
    
    hsv_s = lighting_params.get('hsv_s', -1)
    if hsv_s < 0 or hsv_s > 1:
        return {
            'status': 'error',
            'message': 'HSV Saturation harus antara 0 dan 1'
        }
    
    hsv_v = lighting_params.get('hsv_v', -1)
    if hsv_v < 0 or hsv_v > 1:
        return {
            'status': 'error',
            'message': 'HSV Value harus antara 0 dan 1'
        }
    
    # Validasi contrast
    contrast = lighting_params.get('contrast', [])
    if not contrast or len(contrast) != 2 or contrast[0] >= contrast[1]:
        return {
            'status': 'error',
            'message': 'Range contrast tidak valid'
        }
    
    # Validasi brightness
    brightness = lighting_params.get('brightness', [])
    if not brightness or len(brightness) != 2 or brightness[0] >= brightness[1]:
        return {
            'status': 'error',
            'message': 'Range brightness tidak valid'
        }
    
    # Validasi blur
    blur = lighting_params.get('blur', -1)
    if blur < 0 or blur > 1:
        return {
            'status': 'error',
            'message': 'Blur harus antara 0 dan 1'
        }
    
    # Validasi noise
    noise = lighting_params.get('noise', -1)
    if noise < 0 or noise > 1:
        return {
            'status': 'error',
            'message': 'Noise harus antara 0 dan 1'
        }
    
    # Validasi split
    split_selector = ui_components.get('split_selector')
    split = 'train'  # Default
    if split_selector and hasattr(split_selector, 'children'):
        for child in split_selector.children:
            if hasattr(child, 'children'):
                for grandchild in child.children:
                    if hasattr(grandchild, 'value') and hasattr(grandchild, 'description') and grandchild.description == 'Split:':
                        split = grandchild.value
                        break
    
    if split not in ['train', 'valid', 'test']:
        return {
            'status': 'error',
            'message': 'Split tidak valid'
        }
    
    # Validasi keberadaan dataset
    data_dir = ui_components.get('data_dir', 'data')
    split_dir = os.path.join(data_dir, 'preprocessed', split)
    
    if not os.path.exists(split_dir):
        return {
            'status': 'error',
            'message': f'Dataset {split} tidak ditemukan di {split_dir}'
        }
    
    # Validasi keberadaan gambar
    images_dir = os.path.join(split_dir, 'images')
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        return {
            'status': 'error',
            'message': f'Tidak ada gambar di dataset {split}'
        }
    
    # Validasi keberadaan label
    labels_dir = os.path.join(split_dir, 'labels')
    if not os.path.exists(labels_dir) or not os.listdir(labels_dir):
        return {
            'status': 'error',
            'message': f'Tidak ada label di dataset {split}'
        }
    
    # Semua validasi berhasil
    return {
        'status': 'success',
        'message': 'Parameter augmentasi valid'
    }

def map_ui_to_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Petakan nilai dari UI ke konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary konfigurasi yang diupdate
    """
    from smartcash.ui.dataset.augmentation.handlers.config_handler import update_config_from_ui
    return update_config_from_ui(ui_components, config)

def map_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Petakan nilai dari konfigurasi ke UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
    """
    from smartcash.ui.dataset.augmentation.handlers.config_handler import update_ui_from_config
    update_ui_from_config(ui_components, config)
