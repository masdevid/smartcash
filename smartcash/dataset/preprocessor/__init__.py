"""
File: smartcash/dataset/preprocessor/__init__.py
Deskripsi: Modul preprocessing untuk normalisasi dan validasi dataset YOLOv5
"""

from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from pathlib import Path
import os
import numpy as np

from .service import PreprocessingService
from .core.engine import PreprocessingEngine, PreprocessingValidator
from .utils import (
    validate_preprocessing_config,
    get_default_preprocessing_config,
    ProgressBridge,
    FileProcessor,
    FileScanner,
    FilenameManager,
    PathResolver,
    CleanupManager
)

# Inisialisasi service
def create_preprocessing_service(config: Dict[str, Any] = None, 
                             progress_tracker=None) -> PreprocessingService:
    """Factory untuk membuat PreprocessingService
    
    Args:
        config: Konfigurasi preprocessing (opsional)
        progress_tracker: Objek untuk melacak progress (opsional)
        
    Returns:
        Instance PreprocessingService
    """
    return PreprocessingService(config, progress_tracker)

# Fungsi utama
def preprocess_dataset(config: Dict[str, Any], 
                     target_split: str = "train",
                     progress_tracker = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Pipeline lengkap preprocessing
    
    Args:
        config: Konfigurasi preprocessing
        target_split: Target split yang akan diproses ('train', 'valid', 'test')
        progress_tracker: Objek untuk melacak progress (opsional)
        progress_callback: Fungsi callback untuk update progress (opsional)
        
    Returns:
        Dict berisi hasil preprocessing
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.preprocess_and_visualize(target_split, progress_callback)

def get_preprocessing_samples(config: Dict[str, Any], 
                           target_split: str = "train",
                           max_samples: int = 5,
                           progress_tracker = None) -> Dict[str, Any]:
    """Ambil sampel acak untuk evaluasi
    
    Args:
        config: Konfigurasi preprocessing
        target_split: Target split yang akan diambil sampelnya
        max_samples: Jumlah maksimal sampel yang diambil
        progress_tracker: Objek untuk melacak progress (opsional)
        
    Returns:
        Dict berisi sampel data
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.get_sampling(target_split, max_samples)

def validate_dataset(config: Dict[str, Any], 
                    target_split: str = "train",
                    progress_tracker = None) -> Dict[str, Any]:
    """Validasi dataset tanpa preprocessing
    
    Args:
        config: Konfigurasi validasi
        target_split: Target split yang akan divalidasi
        progress_tracker: Objek untuk melacak progress (opsional)
        
    Returns:
        Dict berisi hasil validasi
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.validate_dataset_only(target_split)

def cleanup_preprocessed_data(config: Dict[str, Any], 
                           target_split: str = None,
                           progress_tracker = None) -> Dict[str, Any]:
    """Hapus file-file hasil preprocessing
    
    Args:
        config: Konfigurasi
        target_split: Target split yang akan dibersihkan (None untuk semua split)
        progress_tracker: Objek untuk melacak progress (opsional)
        
    Returns:
        Dict berisi status cleanup
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.cleanup_preprocessed_data(target_split)

def get_preprocessing_status(config: Dict[str, Any],
                           progress_tracker = None) -> Dict[str, Any]:
    """Dapatkan status preprocessing
    
    Args:
        config: Konfigurasi
        progress_tracker: Objek untuk melacak progress (opsional)
        
    Returns:
        Dict berisi status preprocessing
    """
    service = create_preprocessing_service(config, progress_tracker)
    return service.get_preprocessing_status()

# Ekspor fungsi-fungsi utama
__all__ = [
    # Kelas utama
    'PreprocessingService',
    'PreprocessingEngine',
    'PreprocessingValidator',
    
    # Fungsi utilitas
    'validate_preprocessing_config',
    'get_default_preprocessing_config',
    'create_preprocessing_service',
    
    # Fungsi utama
    'preprocess_dataset',
    'get_preprocessing_samples',
    'validate_dataset',
    'cleanup_preprocessed_data',
    'get_preprocessing_status',
    
    # Komponen utilitas
    'ProgressBridge',
    'FileProcessor',
    'FileScanner',
    'PathResolver',
    'CleanupManager'
]