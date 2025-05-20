"""
File: smartcash/ui/dataset/preprocessing/handlers/service_handler.py
Deskripsi: Handler untuk interaksi dengan service preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger()

def get_dataset_manager(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None, custom_logger=None) -> Any:
    """
    Dapatkan instance dataset manager untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)
        custom_logger: Logger kustom (opsional)
        
    Returns:
        Instance dataset manager atau None jika gagal
    """
    logger = custom_logger or ui_components.get('logger', get_logger())
    
    try:
        # Import dataset manager
        from smartcash.dataset.services.preprocessing_manager import PreprocessingManager
        
        # Dapatkan path dari UI components
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Pastikan path dalam bentuk string
        data_dir_str = str(data_dir) if isinstance(data_dir, Path) else data_dir
        preprocessed_dir_str = str(preprocessed_dir) if isinstance(preprocessed_dir, Path) else preprocessed_dir
        
        # Buat konfigurasi untuk PreprocessingManager
        preprocessing_config = {
            'preprocessing': {
                'raw_dataset_dir': data_dir_str,
                'preprocessed_dir': preprocessed_dir_str
            }
        }
        
        # Buat instance dataset manager
        dataset_manager = PreprocessingManager(
            config=preprocessing_config,
            logger=logger
        )
        
        # Update konfigurasi jika tersedia
        if config and hasattr(dataset_manager, 'config'):
            # Update konfigurasi dataset
            if 'data' in config:
                dataset_manager.config.update({
                    'dataset_dir': data_dir_str,
                    'preprocessed_dir': preprocessed_dir_str
                })
            
            # Update konfigurasi preprocessing
            if 'preprocessing' in config:
                dataset_manager.config['preprocessing'] = config.get('preprocessing', {})
        
        # Log info
        logger.info(f"{ICONS['info']} Dataset manager berhasil dibuat dengan direktori:")
        logger.info(f"  - Data: {data_dir_str}")
        logger.info(f"  - Preprocessed: {preprocessed_dir_str}")
        
        return dataset_manager
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat membuat dataset manager: {str(e)}")
        return None

def run_preprocessing(ui_components: Dict[str, Any], params: Dict[str, Any]) -> bool:
    """
    Jalankan proses preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter preprocessing
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan dataset manager
        dataset_manager = get_dataset_manager(ui_components)
        if not dataset_manager:
            logger.error(f"{ICONS['error']} Gagal mendapatkan dataset manager")
            return False
        
        # Validasi parameter
        from smartcash.ui.dataset.preprocessing.handlers.parameter_handler import validate_preprocessing_params
        validated_params = validate_preprocessing_params(params)
        
        # Setup progress tracking
        setup_progress_tracking(ui_components)
        
        # Jalankan preprocessing
        result = dataset_manager.preprocess_dataset(**validated_params)
        
        # Update UI setelah preprocessing
        update_ui_after_preprocessing(ui_components, result)
        
        return result.get('success', False)
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat menjalankan preprocessing: {str(e)}")
        return False

def setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """
    Setup progress tracking untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Dapatkan komponen progress
    progress_bar = ui_components.get('progress_bar')
    current_progress = ui_components.get('current_progress')
    overall_label = ui_components.get('overall_label')
    step_label = ui_components.get('step_label')
    
    # Tampilkan komponen progress
    if progress_bar:
        progress_bar.value = 0
        progress_bar.layout.visibility = 'visible'
    
    if current_progress:
        current_progress.value = 0
        current_progress.layout.visibility = 'visible'
    
    if overall_label:
        overall_label.value = "Memulai preprocessing..."
        overall_label.layout.visibility = 'visible'
    
    if step_label:
        step_label.value = "Menginisialisasi..."
        step_label.layout.visibility = 'visible'

def update_ui_after_preprocessing(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Update UI setelah preprocessing selesai.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil preprocessing
    """
    # Update progress bar
    progress_bar = ui_components.get('progress_bar')
    if progress_bar:
        progress_bar.value = 100
    
    # Update label
    overall_label = ui_components.get('overall_label')
    if overall_label:
        status = "berhasil" if result.get('success', False) else "gagal"
        overall_label.value = f"Preprocessing {status}"
    
    # Tampilkan tombol visualisasi
    visualization_buttons = ui_components.get('visualization_buttons')
    if visualization_buttons and hasattr(visualization_buttons, 'layout'):
        visualization_buttons.layout.display = 'flex'
    
    # Sinkronkan konfigurasi dengan drive
    from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import sync_config_with_drive
    sync_config_with_drive(ui_components)
