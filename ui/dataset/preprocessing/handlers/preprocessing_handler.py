"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handler.py
Deskripsi: Handler untuk tombol preprocessing dataset
"""

from typing import Dict, Any, Optional, Tuple, List
import os
from pathlib import Path
from IPython.display import display

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_state, update_status_panel, update_ui_before_preprocessing, 
    is_preprocessing_running, set_preprocessing_state, show_confirmation
)
from smartcash.ui.dataset.preprocessing.utils.progress_manager import (
    update_progress, reset_progress_bar, start_progress, complete_progress, create_progress_callback
)
from smartcash.ui.dataset.preprocessing.utils.ui_observers import (
    notify_process_start, notify_process_complete, notify_process_error
)
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

def handle_preprocessing_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol preprocessing dataset.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Cek apakah preprocessing sudah berjalan
        if is_preprocessing_running(ui_components):
            log_message(ui_components, "Preprocessing sudah berjalan", "warning", "⚠️")
            return
            
        # Get config dari UI
        config = get_preprocessing_config_from_ui(ui_components)
        
        # Log mulai preprocessing
        log_message(ui_components, "Memulai preprocessing dataset...", "info", "🔄")
        
        # Update UI state
        update_status_panel(ui_components, "warning", "Konfirmasi preprocessing dataset...")
        
        # Tampilkan dialog konfirmasi
        confirm_preprocessing(ui_components, config, button)
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat persiapan preprocessing: {error_message}")
        log_message(ui_components, f"Error saat persiapan preprocessing: {error_message}", "error", "❌")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def get_preprocessing_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi preprocessing dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi preprocessing
    """
    config = {}
    
    # Get preprocessing options
    if 'preprocess_options' in ui_components:
        options = ui_components['preprocess_options']
        
        # Resolution
        if hasattr(options, 'resolution') and hasattr(options.resolution, 'value'):
            resolution = options.resolution.value
            # Parse resolution string ke tuple jika dalam format WxH
            if isinstance(resolution, str) and 'x' in resolution:
                try:
                    width, height = resolution.split('x')
                    config['resolution'] = (int(width), int(height))
                except ValueError:
                    config['resolution'] = resolution
            else:
                config['resolution'] = resolution
        
        # Normalization
        if hasattr(options, 'normalization') and hasattr(options.normalization, 'value'):
            config['normalization'] = 'minmax' if options.normalization.value else 'none'
        
        # Workers
        if hasattr(options, 'num_workers') and hasattr(options.num_workers, 'value'):
            config['num_workers'] = options.num_workers.value
            
        # Target Split
        if hasattr(options, 'target_split') and hasattr(options.target_split, 'value'):
            split_value = options.target_split.value
            split_map = {
                'Train': 'train',
                'Validation': 'val',
                'Test': 'test',
                'All': 'all'
            }
            config['split'] = split_map.get(split_value, 'all')
    
    # Compatibility fallback untuk split dari widget split_selector
    if 'split' not in config and 'split_selector' in ui_components and hasattr(ui_components['split_selector'], 'value'):
        split_value = ui_components['split_selector'].value
        split_map = {
            'Train Only': 'train',
            'Validation Only': 'val',
            'Test Only': 'test',
            'All Splits': 'all'
        }
        config['split'] = split_map.get(split_value, 'all')
    
    # Get validation options
    if 'validation_options' in ui_components:
        validation_options = ui_components['validation_options']
        if hasattr(validation_options, 'get_selected') and callable(validation_options.get_selected):
            try:
                config['validation_items'] = validation_options.get_selected()
            except (AttributeError, TypeError):
                # Default validation items jika method tidak tersedia
                config['validation_items'] = ['validate_image_format', 'validate_label_format']
    
    # Get data directories
    config['data_dir'] = ui_components.get('data_dir', 'data')
    config['preprocessed_dir'] = ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    return config

def confirm_preprocessing(ui_components: Dict[str, Any], config: Dict[str, Any], button: Any = None) -> None:
    """
    Tampilkan dialog konfirmasi untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
        button: Tombol yang diklik
    """
    # Format pesan konfirmasi
    resolution = config.get('resolution', 'default')
    if isinstance(resolution, tuple) and len(resolution) == 2:
        resolution_str = f"{resolution[0]}x{resolution[1]}"
    else:
        resolution_str = str(resolution)
    
    normalization = config.get('normalization', 'default')
    augmentation = "Ya" if config.get('augmentation', False) else "Tidak"
    split = config.get('split', 'all')
    split_map = {
        'train': 'Training',
        'val': 'Validasi',
        'test': 'Testing',
        'all': 'Semua'
    }
    split_str = split_map.get(split, 'Semua')
    
    message = f"Anda akan menjalankan preprocessing dataset dengan konfigurasi:\n\n"
    message += f"• Resolusi: {resolution_str}\n"
    message += f"• Normalisasi: {normalization}\n"
    message += f"• Augmentasi: {augmentation}\n"
    message += f"• Split: {split_str}\n\n"
    message += "Apakah Anda yakin ingin melanjutkan?"
    
    # Fungsi untuk menjalankan preprocessing dan membersihkan dialog
    def confirm_and_execute():
        # Jalankan preprocessing
        execute_preprocessing(ui_components, config)
    
    # Fungsi untuk membatalkan preprocessing
    def cancel_preprocessing(_=None):
        log_message(ui_components, "Preprocessing dibatalkan", "info", "ℹ️")
        update_status_panel(ui_components, "info", "Preprocessing dibatalkan")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False
    
    # Gunakan fungsi konfirmasi dari ui_state_manager
    show_confirmation(
        ui_components=ui_components,
        title="Konfirmasi Preprocessing Dataset",
        message=message,
        on_confirm=confirm_and_execute,
        on_cancel=cancel_preprocessing
    )

def execute_preprocessing(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Eksekusi preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
    """
    # Update UI status
    update_ui_before_preprocessing(ui_components)
    
    # Set flag bahwa preprocessing sedang berjalan
    set_preprocessing_state(ui_components, True)
    
    # Tampilkan progress bar
    start_progress(ui_components, "Mempersiapkan preprocessing dataset...")
    
    try:
        # Import komponen yang dibutuhkan
        from smartcash.common.config import get_config_manager
        from smartcash.dataset.services.preprocessor.preprocessing_service import PreprocessingService
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Dapatkan parameter
        split = config.get('split', 'all')
        normalization = config.get('normalization', 'minmax')
        augmentation = config.get('augmentation', False)
        num_workers = config.get('num_workers', 4)
        
        # Dapatkan resolusi
        resolution = config.get('resolution')
        if isinstance(resolution, tuple) and len(resolution) == 2:
            img_size = resolution
        elif isinstance(resolution, str) and 'x' in resolution:
            try:
                width, height = map(int, resolution.split('x'))
                img_size = (width, height)
            except ValueError:
                img_size = (640, 640) # Default size
        else:
            img_size = (640, 640) # Default size
        
        # Siapkan config untuk DatasetPreprocessor
        preprocessor_config = {
            'preprocessing': {
                'img_size': img_size,
                'normalize': normalization != 'none',
                'normalization': normalization,
                'augmentation': augmentation,
                'num_workers': num_workers,
                'output_dir': config.get('preprocessed_dir', 'data/preprocessed')
            },
            'data': {
                'dir': config.get('data_dir', 'data')
            }
        }
        
        # Log parameters
        log_message(ui_components, f"Preprocessing dataset dengan resolusi {img_size}, normalisasi {normalization}, augmentasi {augmentation}, split {split}, workers {num_workers}", "info", "🔄")
        
        # Notify process start
        notify_process_start(ui_components, "preprocessing", f"split: {split}")
        
        # Update progress
        update_progress(ui_components, 10, 100, "Memulai preprocessing dataset...", f"Menganalisis dataset untuk split: {split}")
        
        # Buat preprocessing service dengan observer_manager dari UI
        preprocessing_service = PreprocessingService(
            config=preprocessor_config,
            logger=ui_components.get('logger'),
            observer_manager=ui_components.get('observer_manager')
        )
        
        # Dapatkan preprocessor dan register progress callback
        preprocessor = preprocessing_service.preprocessor
        
        # Buat dan register progress callback
        progress_callback = create_progress_callback(ui_components)
        preprocessor.register_progress_callback(progress_callback)
        
        # Jalankan preprocessing
        result = preprocessing_service.preprocess_dataset(
            split=split,
            force_reprocess=config.get('force_reprocess', False)
        )
        
        # Update progress
        complete_progress(ui_components, "Preprocessing selesai")
        
        # Process result
        if result.get('success', False):
            # Get statistics
            total_images = result.get('total_images', 0)
            total_skipped = result.get('total_skipped', 0)
            total_failed = result.get('total_failed', 0)
            
            # Update UI success
            update_ui_state(ui_components, "success", "Preprocessing berhasil diselesaikan")
            
            # Log success
            log_message(ui_components, f"Preprocessing berhasil: {total_images} gambar diproses, {total_skipped} gambar dilewati, {total_failed} gambar gagal", "success", "✅")
            
            # Notify completion
            notify_process_complete(ui_components, result, f"split: {split}")
        else:
            error_message = result.get('error', 'Unknown error')
            
            # Update UI error
            update_ui_state(ui_components, "error", f"Preprocessing gagal: {error_message}")
            
            # Log error
            log_message(ui_components, f"Preprocessing gagal: {error_message}", "error", "❌")
            
            # Notify error
            notify_process_error(ui_components, error_message)
    
    except Exception as e:
        error_message = str(e)
        
        # Update UI error
        update_ui_state(ui_components, "error", f"Error saat preprocessing: {error_message}")
        
        # Log error
        log_message(ui_components, f"Error saat preprocessing: {error_message}", "error", "❌")
        
        # Notify error
        notify_process_error(ui_components, error_message)
    
    finally:
        # Reset preprocessing state
        set_preprocessing_state(ui_components, False) 