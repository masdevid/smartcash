"""
File: smartcash/ui/dataset/augmentation/handlers/initialization_handler.py
Deskripsi: Handler untuk inisialisasi proses augmentasi dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger("initialization_handler")

def initialize_augmentation_directories(
    ui_components: Dict[str, Any], 
    split: str = 'train'
) -> Dict[str, Any]:
    """
    Inisialisasi dan validasi direktori yang diperlukan untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset yang akan diaugmentasi
        
    Returns:
        Dictionary dengan informasi direktori dan status sukses
    """
    try:
        # Dapatkan direktori dari ui_components atau gunakan default
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Cek apakah direktori data ada
        if not os.path.exists(data_dir):
            return {
                'success': False,
                'message': f"Direktori data tidak ditemukan: {data_dir}"
            }
        
        # Validasi direktori input
        input_dir = Path(data_dir) / 'images'
        if not input_dir.exists():
            # Coba cari di preprocessed_dir jika ada
            if os.path.exists(preprocessed_dir):
                input_dir = Path(preprocessed_dir) / split / 'images'
                if not input_dir.exists():
                    input_dir = Path(preprocessed_dir) / 'images'
                    if not input_dir.exists():
                        input_dir = Path(preprocessed_dir)
            else:
                # Fallback ke data_dir
                input_dir = Path(data_dir)
        
        # Validasi direktori output
        output_dir = Path(augmented_dir)
        if not output_dir.exists():
            os.makedirs(str(output_dir), exist_ok=True)
            logger.info(f"{ICONS['info']} Membuat direktori output: {output_dir}")
        
        # Cek apakah ada gambar di direktori input
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(input_dir.glob(f"*{ext}")))
        
        if not image_files:
            return {
                'success': False,
                'message': f"Tidak ada gambar di direktori input: {input_dir}"
            }
        
        # Validasi direktori labels jika ada
        labels_dir = input_dir.parent / 'labels'
        has_labels = labels_dir.exists()
        
        return {
            'success': True,
            'message': f"Berhasil menginisialisasi direktori dengan {len(image_files)} gambar",
            'data_dir': str(data_dir),
            'preprocessed_dir': str(preprocessed_dir),
            'augmented_dir': str(augmented_dir),
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'has_labels': has_labels,
            'image_count': len(image_files)
        }
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat inisialisasi direktori: {str(e)}")
        return {
            'success': False,
            'message': f"Error saat inisialisasi direktori: {str(e)}"
        }

def validate_augmentation_prerequisites(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi prasyarat sebelum memulai augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary status validasi dengan format yang konsisten
    """
    # Validasi direktori dengan initialize_augmentation_directories
    init_result = initialize_augmentation_directories(ui_components)
    if not init_result.get('success', False):
        return {
            'success': False,
            'message': init_result.get('message', "Gagal inisialisasi direktori")
        }
    
    # Validasi parameter augmentasi
    try:
        # Import handler untuk konfigurasi augmentasi
        from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import get_augmentation_config
        aug_config = get_augmentation_config(ui_components)
        
        # Validasi parameter dasar
        if not aug_config.get('enabled', True):
            return {
                'success': False,
                'message': "Augmentasi tidak diaktifkan"
            }
        
        # Validasi jumlah gambar per input
        if aug_config.get('num_per_image', 0) <= 0:
            return {
                'success': False,
                'message': "Jumlah gambar per input harus lebih dari 0"
            }
        
        # Validasi jenis augmentasi yang diaktifkan
        augmentation_types = ['geometric', 'color', 'noise', 'cutout', 'mosaic', 'mixup']
        if not any(aug_config.get(aug_type, False) for aug_type in augmentation_types):
            return {
                'success': False,
                'message': "Tidak ada jenis augmentasi yang diaktifkan"
            }
        
        # Tambahkan informasi gambar dan konfigurasi
        return {
            'success': True,
            'message': f"Validasi berhasil: {init_result.get('image_count', 0)} gambar siap diaugmentasi",
            'image_count': init_result.get('image_count', 0),
            'input_dir': init_result.get('input_dir'),
            'output_dir': init_result.get('augmented_dir'),
            'aug_config': aug_config
        }
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat validasi parameter augmentasi: {str(e)}")
        return {
            'success': False,
            'message': f"Error saat validasi parameter: {str(e)}"
        }
