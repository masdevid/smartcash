"""
File: smartcash/ui/dataset/preprocessing/handlers/button_handler.py
Deskripsi: Handler tombol untuk modul preprocessing dataset
"""

from typing import Dict, Any, Optional, Callable, Union, List
from IPython.display import display, clear_output
import os
import traceback
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
from smartcash.ui.dataset.preprocessing.utils.ui_observers import (
    notify_process_start, 
    notify_process_complete, 
    notify_process_error, 
    notify_process_stop,
    disable_ui_during_processing
)

def execute_preprocessing(ui_components: Dict[str, Any], split, split_info: str):
    """Eksekusi preprocessing dengan parameter dari UI."""
    logger = ui_components.get('logger')
    
    # Dapatkan notification manager
    from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager
    notification_manager = get_notification_manager(ui_components)
    
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
        
        # Register progress callback jika tersedia
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            dataset_manager.register_progress_callback(ui_components['update_progress'])
        
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
        notification_manager.update_status("info", f"{ICONS['processing']} Menjalankan preprocessing untuk {split_info}...")
        
        # Set flag running
        ui_components['preprocessing_running'] = True
        
        # Jalankan preprocessing
        result = dataset_manager.preprocess(
            split=split,
            config=config,
            stop_flag=lambda: not ui_components.get('preprocessing_running', True)
        )
        
        # Update UI setelah selesai
        if ui_components.get('preprocessing_running', True):
            # Preprocessing selesai dengan normal
            notification_manager.update_status("success", f"{ICONS['success']} Preprocessing {split_info} selesai")
            
            # Notifikasi observer
            notify_process_complete(ui_components, result, split_info)
        else:
            # Preprocessing dihentikan oleh pengguna
            notification_manager.update_status("warning", f"{ICONS['warning']} Preprocessing {split_info} dihentikan")
        
        # Aktifkan kembali UI
        disable_ui_during_processing(ui_components, False)
        
        # Tampilkan tombol primary dan sembunyikan tombol stop
        if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'layout'):
            ui_components['preprocess_button'].layout.display = 'block'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
        
        # Reset flag
        ui_components['preprocessing_running'] = False
        
        return result
    
    except Exception as e:
        # Tangkap error
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        # Log error
        if logger:
            logger.error(f"{ICONS['error']} Error saat preprocessing {split_info}: {error_msg}")
            logger.debug(f"Stack trace: {stack_trace}")
        
        # Update status
        notification_manager.update_status("error", f"{ICONS['error']} Error: {error_msg}")
        
        # Notifikasi observer
        notify_process_error(ui_components, error_msg)
        
        # Aktifkan kembali UI
        disable_ui_during_processing(ui_components, False)
        
        # Tampilkan tombol primary dan sembunyikan tombol stop
        if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'layout'):
            ui_components['preprocess_button'].layout.display = 'block'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
        
        # Reset flag
        ui_components['preprocessing_running'] = False

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
    logger = ui_components.get('logger', get_logger(module_type))
    
    # Dapatkan notification manager
    from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager
    notification_manager = get_notification_manager(ui_components)
    
    # Set flag running ke False
    ui_components['preprocessing_running'] = False
    
    # Handler tombol primary dengan dukungan progress tracking
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
        split_info = split_value
        
        # Set flag running
        ui_components['preprocessing_running'] = True
        
        # Disable UI selama proses
        disable_ui_during_processing(ui_components, True)
        
        # Sembunyikan tombol primary dan tampilkan tombol stop
        if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'layout'):
            ui_components['preprocess_button'].layout.display = 'none'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'block'
        
        # Update status
        notification_manager.update_status("info", f"{ICONS['processing']} Memulai preprocessing {split_info}...")
        
        # Notifikasi observer bahwa proses dimulai
        notify_process_start(ui_components, "preprocessing", split_info, split)
        
        # Jalankan preprocessing dengan parameter dari UI
        execute_preprocessing(ui_components, split, split_info)
    
    # Handler untuk menghentikan proses
    def on_stop_click(b):
        # Set flag untuk menghentikan proses
        ui_components['preprocessing_running'] = False
        
        # Update status
        notification_manager.update_status("warning", f"{ICONS['warning']} Menghentikan preprocessing...")
        
        # Notifikasi observer bahwa proses dihentikan
        notify_process_stop(ui_components)
    
    # Reset UI dan konfigurasi ke default
    def on_reset_click(b):
        # Konfirmasi reset
        notification_manager.update_status("warning", f"{ICONS['warning']} Reset konfigurasi preprocessing...")
        
        # Reset UI ke kondisi awal
        _reset_ui(ui_components)
        
        # Update status
        notification_manager.update_status("info", f"{ICONS['info']} Konfigurasi preprocessing direset ke default")
    
    # Reset UI ke kondisi awal
    def _reset_ui(ui_components: Dict[str, Any]):
        # Reset flag
        ui_components['preprocessing_running'] = False
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        
        # Reset UI components ke default
        # Dapatkan parameter dari preprocess_options
        preproc_options = ui_components.get('preprocess_options')
        if preproc_options and hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
            preproc_options.children[0].value = 640  # Image size
            preproc_options.children[1].value = True  # Normalize
            preproc_options.children[2].value = True  # Preserve Aspect Ratio
            preproc_options.children[3].value = True  # Cache enabled
            preproc_options.children[4].value = 4     # Workers
        
        # Reset validation options
        validation_options = ui_components.get('validation_options')
        if validation_options and hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
            validation_options.children[0].value = True  # Enable validation
            validation_options.children[1].value = True  # Fix issues
            validation_options.children[2].value = True  # Move invalid
            validation_options.children[3].value = 'data/invalid'  # Invalid dir
        
        # Aktifkan kembali UI
        disable_ui_during_processing(ui_components, False)
        
        # Tampilkan tombol primary dan sembunyikan tombol stop
        if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'layout'):
            ui_components['preprocess_button'].layout.display = 'block'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
    
    # Register handler untuk tombol
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].on_click(on_primary_click)
    
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(on_stop_click)
    
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(on_reset_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_primary_click': on_primary_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'execute_preprocessing': execute_preprocessing
    })
    
    # Set flag running ke False
    ui_components['preprocessing_running'] = False
    
    return ui_components
