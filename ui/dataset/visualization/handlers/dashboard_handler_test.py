"""
File: smartcash/ui/dataset/visualization/handlers/dashboard_handler_test.py
Deskripsi: Fungsi-fungsi untuk pengujian dashboard visualisasi dataset
"""

import os
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import glob

from smartcash.common.logger import get_logger
from smartcash.common.config.manager import ConfigManager

logger = get_logger(__name__)

def get_dataset_stats(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset.
    
    Args:
        dataset_path: Path ke dataset (opsional)
        
    Returns:
        Dictionary berisi statistik dataset
    """
    if not dataset_path:
        # Coba dapatkan path dari konfigurasi
        config_manager = ConfigManager()
        dataset_config = config_manager.get_config('dataset_config')
        dataset_path = dataset_config.get('dataset_path', None)
        
        if not dataset_path:
            logger.warning("⚠️ Dataset path tidak ditemukan dalam konfigurasi")
            return {
                'split_stats': {
                    'train': {'images': 0, 'labels': 0, 'objects': 0},
                    'val': {'images': 0, 'labels': 0, 'objects': 0},
                    'test': {'images': 0, 'labels': 0, 'objects': 0}
                }
            }
    
    # Hitung jumlah gambar dan label untuk setiap split
    split_stats = {}
    for split in ['train', 'val', 'test']:
        # Hitung jumlah gambar
        images_path = os.path.join(dataset_path, 'images', split)
        images_count = len(glob.glob(os.path.join(images_path, '*.jpg'))) if os.path.exists(images_path) else 0
        
        # Hitung jumlah label
        labels_path = os.path.join(dataset_path, 'labels', split)
        labels_count = len(glob.glob(os.path.join(labels_path, '*.txt'))) if os.path.exists(labels_path) else 0
        
        # Hitung jumlah objek (baca setiap file label)
        objects_count = 0
        if os.path.exists(labels_path):
            for label_file in glob.glob(os.path.join(labels_path, '*.txt')):
                try:
                    with open(label_file, 'r') as f:
                        objects_count += len(f.readlines())
                except Exception as e:
                    logger.warning(f"⚠️ Error saat membaca file label {label_file}: {str(e)}")
        
        # Tambahkan ke statistik
        split_stats[split] = {
            'images': images_count,
            'labels': labels_count,
            'objects': objects_count
        }
    
    return {
        'split_stats': split_stats
    }

def get_preprocessing_stats(dataset_path: str) -> Dict[str, Any]:
    """
    Mendapatkan statistik preprocessing.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Dictionary berisi statistik preprocessing
    """
    # Cek file metadata preprocessing
    metadata_path = os.path.join(dataset_path, 'metadata', 'preprocessing')
    processed_images_file = os.path.join(metadata_path, 'processed_images.txt')
    
    processed_images = 0
    if os.path.exists(processed_images_file):
        try:
            with open(processed_images_file, 'r') as f:
                processed_images = len(f.readlines())
        except Exception as e:
            logger.warning(f"⚠️ Error saat membaca file metadata preprocessing: {str(e)}")
    
    return {
        'processed_images': processed_images
    }

def get_augmentation_stats(dataset_path: str) -> Dict[str, Any]:
    """
    Mendapatkan statistik augmentasi.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Dictionary berisi statistik augmentasi
    """
    # Cek file metadata augmentasi
    metadata_path = os.path.join(dataset_path, 'metadata', 'augmentation')
    augmented_images_file = os.path.join(metadata_path, 'augmented_images.txt')
    
    augmented_images = 0
    if os.path.exists(augmented_images_file):
        try:
            with open(augmented_images_file, 'r') as f:
                augmented_images = len(f.readlines())
        except Exception as e:
            logger.warning(f"⚠️ Error saat membaca file metadata augmentasi: {str(e)}")
    
    return {
        'augmented_images': augmented_images
    }

def get_processing_status(dataset_path: str) -> Dict[str, Any]:
    """
    Mendapatkan status pemrosesan dataset.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Dictionary berisi status pemrosesan
    """
    # Inisialisasi status
    preprocessing_status = {
        'train': False,
        'val': False,
        'test': False
    }
    
    augmentation_status = {
        'train': False,
        'val': False,
        'test': False
    }
    
    # Cek file status preprocessing
    metadata_path = os.path.join(dataset_path, 'metadata', 'preprocessing')
    for split in ['train', 'val', 'test']:
        status_file = os.path.join(metadata_path, f'{split}_processed.txt')
        if os.path.exists(status_file):
            preprocessing_status[split] = True
    
    # Cek file status augmentasi
    metadata_path = os.path.join(dataset_path, 'metadata', 'augmentation')
    for split in ['train', 'val', 'test']:
        status_file = os.path.join(metadata_path, f'{split}_augmented.txt')
        if os.path.exists(status_file):
            augmentation_status[split] = True
    
    return {
        'preprocessing_status': preprocessing_status,
        'augmentation_status': augmentation_status
    }
