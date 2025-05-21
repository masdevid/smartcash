"""
File: smartcash/ui/dataset/preprocessing/handlers/button_handler.py
Deskripsi: Handler tombol untuk modul preprocessing dataset
"""

from typing import Dict, Any, Optional, Callable, Union, List
from IPython.display import display, clear_output
import os
import traceback
from pathlib import Path

# Import utils dari preprocessing module
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_status_panel,
    update_ui_before_preprocessing,
    reset_ui_after_preprocessing, 
    toggle_input_controls
)
from smartcash.ui.dataset.preprocessing.utils.progress_manager import (
    start_progress,
    update_progress,
    complete_progress,
    reset_progress_bar
)
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    PREPROCESSING_LOGGER_NAMESPACE,
    notify_log,
    notify_progress
)
from smartcash.ui.dataset.preprocessing.handlers.observer_handler import (
    notify_process_start,
    notify_process_complete,
    notify_process_error,
    notify_process_progress
)
from smartcash.ui.dataset.preprocessing.handlers.confirmation_handler import (
    confirm_preprocessing
)

# Import common
from smartcash.common.logger import get_logger

# Setup logger
logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)

def execute_preprocessing(ui_components: Dict[str, Any], split, split_info: str):
    """Eksekusi preprocessing dengan parameter dari UI."""
    
    # Dapatkan parameter dari UI
    try:
        # Dapatkan dataset manager
        dataset_manager = ui_components.get('dataset_manager')
        if not dataset_manager:
            from smartcash.dataset.services.preprocessor.preprocessing_service import PreprocessingService
            dataset_manager = PreprocessingService(
                config=ui_components.get('config', {}),
                logger=logger
            )
            ui_components['dataset_manager'] = dataset_manager
        
        # Register progress callback ke dataset manager
        dataset_manager.register_progress_callback(
            lambda progress, total, message: notify_process_progress(ui_components, progress, total, message)
        )
        
        # Dapatkan parameter preprocessing dari UI
        params = {}
        
        # Dapatkan parameter dari preprocess_options
        preproc_options = ui_components.get('preprocess_options')
        if preproc_options and hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
            img_size = preproc_options.children[0].value
            normalize = preproc_options.children[1].value
            preserve_aspect_ratio = preproc_options.children[2].value
            cache_enabled = preproc_options.children[3].value
            num_workers = preproc_options.children[4].value
            
            params.update({
                'img_size': [img_size, img_size],  # Square aspect ratio
                'normalization': {
                    'enabled': normalize,
                    'preserve_aspect_ratio': preserve_aspect_ratio
                },
                'enabled': cache_enabled,
                'num_workers': num_workers
            })
        
        # Dapatkan parameter dari validation_options
        validation_options = ui_components.get('validation_options')
        if validation_options and hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
            validate_enabled = validation_options.children[0].value
            fix_issues = validation_options.children[1].value
            move_invalid = validation_options.children[2].value
            invalid_dir = validation_options.children[3].value
            
            params.update({
                'validate': {
                    'enabled': validate_enabled,
                    'fix_issues': fix_issues,
                    'move_invalid': move_invalid,
                    'invalid_dir': invalid_dir
                }
            })
        
        # Dapatkan config dari ui_components
        config = ui_components.get('config', {})
        
        # Update config dengan parameter dari UI
        if 'preprocessing' not in config:
            config['preprocessing'] = {}
        
        config['preprocessing'].update(params)
        
        # Update status
        update_status_panel(ui_components, "started", f"Menjalankan preprocessing untuk {split_info}...")
        
        # Set UI untuk running
        update_ui_before_preprocessing(ui_components)
        
        # Start progress tracking
        start_progress(ui_components, f"Memulai preprocessing {split_info}...")
        
        # Notifikasi observer
        notify_process_start(ui_components, "preprocessing", split_info, split)
        
        # Jalankan preprocessing
        result = dataset_manager.preprocess(
            split=split,
            config=config,
            stop_flag=lambda: not ui_components.get('preprocessing_running', True)
        )
        
        # Update UI setelah selesai
        if ui_components.get('preprocessing_running', True):
            # Preprocessing selesai dengan normal
            update_status_panel(ui_components, "completed", f"Preprocessing {split_info} selesai")
            
            # Complete progress tracking
            complete_progress(ui_components, f"Preprocessing {split_info} selesai")
            
            # Notifikasi observer
            notify_process_complete(ui_components, result, split_info)
        else:
            # Preprocessing dihentikan oleh pengguna
            update_status_panel(ui_components, "warning", f"Preprocessing {split_info} dihentikan")
            
            # Reset progress tracking
            reset_progress_bar(ui_components)
        
        # Reset UI setelah preprocessing
        reset_ui_after_preprocessing(ui_components)
        
        return result
    
    except Exception as e:
        # Tangkap error
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        # Log error
        log_message(ui_components, f"Error saat preprocessing {split_info}: {error_msg}", "error", "❌")
        
        # Untuk debugging, log stack trace 
        logger.debug(f"Stack trace: {stack_trace}")
        
        # Update status
        update_status_panel(ui_components, "failed", f"Error: {error_msg}")
        
        # Notifikasi observer
        notify_process_error(ui_components, error_msg)
        
        # Reset UI setelah preprocessing
        reset_ui_after_preprocessing(ui_components)

def setup_preprocessing_button_handlers(
    ui_components: Dict[str, Any], 
    module_type: str = 'preprocessing',
    config: Dict[str, Any] = None, 
    env = None
) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul (default: 'preprocessing')
        config: Konfigurasi modul (opsional)
        env: Environment manager (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Set flag running ke False
    ui_components['preprocessing_running'] = False
    
    # Handler tombol primary dengan dukungan progress tracking dan konfirmasi
    def on_primary_click(b):
        # Dapatkan split yang dipilih
        split_selector = ui_components.get('split_selector')
        split_value = split_selector.value if split_selector else "All Splits"
        
        # Map dari UI value ke split value
        split_map = {
            'All Splits': None,  # None berarti semua split
            'Train Only': 'train',
            'Validation Only': 'valid',
            'Test Only': 'test'
        }
        
        split = split_map.get(split_value, None)
        
        # Disable inputs selama preprocessing
        toggle_input_controls(ui_components, True)
        
        # Tampilkan konfirmasi terlebih dahulu
        confirm_preprocessing(ui_components, split)
    
    # Handler tombol stop
    def on_stop_click(b):
        # Set flag untuk menghentikan proses
        ui_components['preprocessing_running'] = False
        
        # Update status
        update_status_panel(ui_components, "warning", "Menghentikan preprocessing...")
        
        # Log ke console
        log_message(ui_components, "Menghentikan preprocessing...", "warning", "⚠️")
    
    # Handler tombol reset
    def on_reset_click(b):
        # Reset konfigurasi ke default
        if 'reset_preprocessing_config' in ui_components and callable(ui_components['reset_preprocessing_config']):
            ui_components['reset_preprocessing_config'](ui_components)
        
        # Reset UI
        _reset_ui(ui_components)
    
    def _reset_ui(ui_components: Dict[str, Any]):
        # Reset flag
        ui_components['preprocessing_running'] = False
        
        # Reset progress bar
        reset_progress_bar(ui_components)
        
        # Reset UI komponen
        reset_ui_after_preprocessing(ui_components)
        
        # Update status
        update_status_panel(ui_components, "idle", "Preprocessing direset")
        
        # Log ke console
        log_message(ui_components, "Preprocessing direset", "info", "🔄")
    
    # Attach event handlers ke tombol
    buttons = {
        'preprocess_button': on_primary_click,
        'stop_button': on_stop_click,
        'reset_button': on_reset_click
    }
    
    for button_key, handler in buttons.items():
        if button_key in ui_components and hasattr(ui_components[button_key], 'on_click'):
            ui_components[button_key].on_click(handler)
    
    # Disable tombol stop awal
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'disabled'):
        ui_components['stop_button'].disabled = True
    
    # Log setup berhasil
    log_message(ui_components, "Button handlers berhasil disetup", "debug", "✅")
    
    return ui_components
