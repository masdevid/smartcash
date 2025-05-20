"""
File: smartcash/ui/dataset/preprocessing/handlers/initialization_handler.py
Deskripsi: Handler untuk inisialisasi proses preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger()

def initialize_preprocessing_directories(
    ui_components: Dict[str, Any], 
    split: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inisialisasi dan validasi direktori yang diperlukan untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset yang akan dipreprocessing (opsional)
        
    Returns:
        Dictionary dengan informasi direktori dan status sukses
    """
    try:
        # Dapatkan direktori dari ui_components atau gunakan default
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Cek apakah direktori data ada
        if not os.path.exists(data_dir):
            return {
                'success': False,
                'message': f"Direktori data tidak ditemukan: {data_dir}"
            }
        
        # Cek apakah path input dan output sama (untuk menghindari masalah)
        if os.path.realpath(data_dir) == os.path.realpath(preprocessed_dir):
            return {
                'success': False,
                'message': f"Path data input dan output sama: {data_dir}, ini akan menyebabkan masalah"
            }
        
        # Validasi direktori input
        input_dir = Path(data_dir) / 'images'
        if not input_dir.exists():
            # Coba cari di data_dir langsung
            input_dir = Path(data_dir)
        
        # Validasi direktori output
        output_dir = Path(preprocessed_dir)
        if not output_dir.exists():
            os.makedirs(str(output_dir), exist_ok=True)
            logger.info(f"{ICONS['info']} Membuat direktori output: {output_dir}")
        
        # Jika split ditentukan, buat direktori split
        if split:
            split_dir = output_dir / split
            if not split_dir.exists():
                os.makedirs(str(split_dir), exist_ok=True)
                logger.info(f"{ICONS['info']} Membuat direktori split: {split_dir}")
            
            # Buat direktori images di dalam split
            split_images_dir = split_dir / 'images'
            if not split_images_dir.exists():
                os.makedirs(str(split_images_dir), exist_ok=True)
                logger.info(f"{ICONS['info']} Membuat direktori images: {split_images_dir}")
            
            # Buat direktori labels di dalam split jika perlu
            split_labels_dir = split_dir / 'labels'
            if not split_labels_dir.exists():
                os.makedirs(str(split_labels_dir), exist_ok=True)
                logger.info(f"{ICONS['info']} Membuat direktori labels: {split_labels_dir}")
        else:
            # Jika tidak ada split, buat direktori images dan labels langsung di output_dir
            images_dir = output_dir / 'images'
            if not images_dir.exists():
                os.makedirs(str(images_dir), exist_ok=True)
                logger.info(f"{ICONS['info']} Membuat direktori images: {images_dir}")
            
            labels_dir = output_dir / 'labels'
            if not labels_dir.exists():
                os.makedirs(str(labels_dir), exist_ok=True)
                logger.info(f"{ICONS['info']} Membuat direktori labels: {labels_dir}")
        
        # Cek apakah ada gambar di direktori input
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(input_dir.glob(f"*{ext}")))
        
        # Jika tidak ada gambar di direktori input, coba cari di subdirektori
        if not image_files and input_dir.exists():
            # Cek apakah ada subdirektori yang mungkin berisi gambar (seperti 'train', 'val', 'test')
            for subdir in ['train', 'val', 'test']:
                subdir_path = input_dir / subdir
                if subdir_path.exists():
                    for ext in ['.jpg', '.jpeg', '.png']:
                        subdir_images = list(subdir_path.glob(f"*{ext}"))
                        if subdir_images:
                            logger.info(f"{ICONS['info']} Menemukan {len(subdir_images)} gambar di subdirektori: {subdir_path}")
                            # Jika ada gambar di subdirektori, gunakan itu sebagai input
                            image_files.extend(subdir_images)
        
        # Jika masih tidak ada gambar, coba cari di direktori parent
        if not image_files and input_dir.parent.exists() and input_dir.parent != input_dir:
            parent_dir = input_dir.parent
            for ext in ['.jpg', '.jpeg', '.png']:
                parent_images = list(parent_dir.glob(f"*{ext}"))
                if parent_images:
                    logger.info(f"{ICONS['info']} Menemukan {len(parent_images)} gambar di direktori parent: {parent_dir}")
                    # Jika ada gambar di direktori parent, gunakan itu sebagai input
                    image_files.extend(parent_images)
                    input_dir = parent_dir  # Update input_dir ke direktori yang berisi gambar
        
        if not image_files:
            return {
                'success': False,
                'message': f"Tidak ada gambar di direktori input: {input_dir}"
            }
        
        # Validasi direktori labels jika ada
        labels_dir = input_dir.parent / 'labels'
        has_labels = labels_dir.exists()
        
        # Tentukan direktori output berdasarkan split
        if split:
            output_images_dir = output_dir / split / 'images'
            output_labels_dir = output_dir / split / 'labels'
        else:
            output_images_dir = output_dir / 'images'
            output_labels_dir = output_dir / 'labels'
        
        return {
            'success': True,
            'message': f"Berhasil menginisialisasi direktori dengan {len(image_files)} gambar",
            'data_dir': str(data_dir),
            'preprocessed_dir': str(preprocessed_dir),
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'output_images_dir': str(output_images_dir),
            'output_labels_dir': str(output_labels_dir),
            'has_labels': has_labels,
            'image_count': len(image_files),
            'split': split
        }
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat inisialisasi direktori: {str(e)}")
        return {
            'success': False,
            'message': f"Error saat inisialisasi direktori: {str(e)}"
        }

def validate_preprocessing_prerequisites(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi prasyarat sebelum memulai preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary status validasi dengan format yang konsisten
    """
    # Dapatkan split dari UI
    split_option = ui_components.get('split_selector', {}).value if 'split_selector' in ui_components else 'All Splits'
    split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
    split = split_map.get(split_option)
    
    # Validasi direktori dengan initialize_preprocessing_directories
    init_result = initialize_preprocessing_directories(ui_components, split)
    if not init_result.get('success', False):
        return {
            'success': False,
            'message': init_result.get('message', "Gagal inisialisasi direktori")
        }
    
    # Validasi parameter preprocessing
    try:
        # Dapatkan konfigurasi preprocessing dari ui_components
        preprocess_config = ui_components.get('config', {}).get('preprocessing', {})
        
        # Jika tidak ada konfigurasi di ui_components, gunakan update_config_from_ui
        if not preprocess_config:
            from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_config_from_ui
            config = update_config_from_ui(ui_components)
            preprocess_config = config.get('preprocessing', {})
        
        # Validasi parameter dasar
        if not preprocess_config.get('enabled', True):
            return {
                'success': False,
                'message': "Preprocessing tidak diaktifkan"
            }
        
        # Validasi jenis preprocessing yang diaktifkan
        preprocessing_types = ['resize', 'normalize', 'augment', 'convert_grayscale', 'denoise']
        if not any(preprocess_config.get(preproc_type, False) for preproc_type in preprocessing_types):
            return {
                'success': False,
                'message': "Tidak ada jenis preprocessing yang diaktifkan"
            }
        
        # Tambahkan informasi gambar dan konfigurasi
        return {
            'success': True,
            'message': f"Validasi berhasil: {init_result.get('image_count', 0)} gambar siap dipreprocessing",
            'image_count': init_result.get('image_count', 0),
            'input_dir': init_result.get('input_dir'),
            'output_dir': init_result.get('preprocessed_dir'),
            'preprocess_config': preprocess_config,
            'split': split
        }
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat validasi parameter preprocessing: {str(e)}")
        return {
            'success': False,
            'message': f"Error saat validasi parameter: {str(e)}"
        }
