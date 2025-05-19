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

# Fungsi notifikasi untuk observer pattern
def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """Notifikasi observer bahwa proses telah dimulai."""
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("preprocessing", {
            'split': split,
            'display_info': display_info
        })

def notify_process_complete(ui_components: Dict[str, Any], result: Dict[str, Any], display_info: str) -> None:
    """Notifikasi observer bahwa proses telah selesai dengan sukses."""
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['success']} Preprocessing {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("preprocessing", result)

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """Notifikasi observer bahwa proses mengalami error."""
    logger = ui_components.get('logger')
    if logger: logger.error(f"{ICONS['error']} Error pada preprocessing: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("preprocessing", error_message)

def notify_process_stop(ui_components: Dict[str, Any], display_info: str = "") -> None:
    """Notifikasi observer bahwa proses telah dihentikan oleh pengguna."""
    logger = ui_components.get('logger')
    if logger: logger.warning(f"{ICONS['stop']} Proses preprocessing dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("preprocessing", {
            'display_info': display_info
        })

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan."""
    # Daftar komponen yang perlu dinonaktifkan
    disable_components = [
        'split_selector', 'config_accordion', 'options_accordion',
        'reset_button', 'preprocess_button', 'save_button'
    ]
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components and hasattr(ui_components[component], 'disabled'):
            ui_components[component].disabled = disable

def execute_preprocessing(ui_components: Dict[str, Any], split, split_info: str):
    """Eksekusi preprocessing dengan parameter dari UI."""
    logger = ui_components.get('logger')
    
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
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](dataset_manager)
        
        # Log awal preprocessing
        if logger: logger.info(f"{ICONS['start']} Memulai preprocessing {split_info}")
        
        # Dapatkan parameter dari UI
        data_dir = ui_components.get('data_dir', '')
        if not data_dir:
            data_dir = ui_components.get('config', {}).get('dataset', {}).get('path', '')
        
        # Dapatkan parameter preprocessing dari UI
        params = {}
        
        # Tambahkan parameter dari UI components
        if 'resize_width' in ui_components:
            params['resize_width'] = ui_components['resize_width'].value
        if 'resize_height' in ui_components:
            params['resize_height'] = ui_components['resize_height'].value
        if 'grayscale' in ui_components:
            params['grayscale'] = ui_components['grayscale'].value
        if 'normalize' in ui_components:
            params['normalize'] = ui_components['normalize'].value
        if 'equalize' in ui_components:
            params['equalize'] = ui_components['equalize'].value
        if 'denoise' in ui_components:
            params['denoise'] = ui_components['denoise'].value
        if 'sharpen' in ui_components:
            params['sharpen'] = ui_components['sharpen'].value
        
        # Jalankan preprocessing
        preprocess_result = dataset_manager.preprocess_dataset(
            data_dir=data_dir,
            split=split,
            **params
        )
        
        # Setelah selesai, update UI dengan status sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS['success']} Preprocessing {split_info} selesai"))
        
        # Update status panel
        update_status_panel(ui_components, "success", 
                           f"{ICONS['success']} Preprocessing dataset berhasil diselesaikan")
        
        # Update UI state - tampilkan summary dan visualisasi
        for component in ['visualization_container', 'summary_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'block'
        
        # Tampilkan tombol visualisasi
        if 'visualization_buttons' in ui_components:
            ui_components['visualization_buttons'].layout.display = 'flex'
        
        # Update summary dengan hasil preprocessing
        if 'generate_summary' in ui_components and callable(ui_components['generate_summary']):
            ui_components['generate_summary'](ui_components)
        
        # Notifikasi observer tentang selesai
        notify_process_complete(ui_components, preprocess_result, split_info)
            
    except Exception as e:
        # Handle error dengan notifikasi
        with ui_components['status']: 
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        
        # Update status panel
        update_status_panel(ui_components, "error", f"{ICONS['error']} Preprocessing gagal: {str(e)}")
        
        # Notifikasi observer tentang error
        notify_process_error(ui_components, str(e))
        
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error saat preprocessing dataset: {str(e)}")
        if logger: logger.error(traceback.format_exc())
    
    finally:
        # Tandai preprocessing selesai
        ui_components['preprocessing_running'] = False
        
        # Restore UI
        if 'cleanup_ui' in ui_components and callable(ui_components['cleanup_ui']):
            ui_components['cleanup_ui'](ui_components)
        else:
            disable_ui_during_processing(ui_components, False)

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
        env: Environment (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger('preprocessing_button'))
    
    # Handler tombol primary dengan dukungan progress tracking
    def on_primary_click(b):
        # Cek apakah sudah running
        if ui_components.get('preprocessing_running', False):
            with ui_components['status']:
                display(create_status_indicator("warning", f"{ICONS['warning']} Preprocessing sudah berjalan"))
            return
        
        # Set flag running
        ui_components['preprocessing_running'] = True
        
        # Disable UI selama proses
        disable_ui_during_processing(ui_components, True)
        
        # Tampilkan tombol stop jika tersedia
        if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'layout'):
            ui_components['preprocess_button'].layout.display = 'none'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'block'
        
        # Expand log accordion jika tersedia
        if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
            ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        
        # Dapatkan split yang dipilih
        split = None
        if 'split_selector' in ui_components:
            split = ui_components['split_selector'].value
        
        # Validasi split
        if not split:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} Error: Split tidak dipilih"))
            
            # Update status panel
            update_status_panel(ui_components, "error", f"{ICONS['error']} Preprocessing gagal: Split tidak dipilih")
            
            # Reset UI
            ui_components['preprocessing_running'] = False
            disable_ui_during_processing(ui_components, False)
            return
        
        # Dapatkan info split untuk display
        split_info = f"untuk split '{split}'"
        
        # Update status
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai preprocessing {split_info}..."))
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai preprocessing {split_info}...")
        
        # Notifikasi observer bahwa proses dimulai
        notify_process_start(ui_components, "preprocessing", split_info, split)
        
        # Jalankan preprocessing dengan parameter dari UI
        execute_preprocessing(ui_components, split, split_info)
    
    # Handler untuk menghentikan proses
    def on_stop_click(b):
        # Set flag untuk menghentikan proses
        ui_components['preprocessing_running'] = False
        
        # Update status
        with ui_components['status']:
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan preprocessing..."))
        
        # Update status panel
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Menghentikan preprocessing...")
        
        # Notifikasi observer bahwa proses dihentikan
        notify_process_stop(ui_components)
    
    # Reset UI dan konfigurasi ke default
    def on_reset_click(b):
        # Konfirmasi reset
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("warning", f"{ICONS['warning']} Reset konfigurasi preprocessing..."))
        
        # Reset UI ke kondisi awal
        _reset_ui(ui_components)
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['info']} Konfigurasi preprocessing direset ke default")
    
    # Reset UI ke kondisi awal
    def _reset_ui(ui_components: Dict[str, Any]):
        # Reset flag
        ui_components['preprocessing_running'] = False
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        
        # Reset UI components ke default
        default_values = {
            'resize_width': 640,
            'resize_height': 640,
            'grayscale': False,
            'normalize': True,
            'equalize': False,
            'denoise': False,
            'sharpen': False
        }
        
        # Update UI components dengan nilai default
        for key, value in default_values.items():
            if key in ui_components:
                if hasattr(ui_components[key], 'value'):
                    ui_components[key].value = value
        
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
        'execute_preprocessing': execute_preprocessing,
        'disable_ui_during_processing': disable_ui_during_processing,
        'notify_process_start': notify_process_start,
        'notify_process_complete': notify_process_complete,
        'notify_process_error': notify_process_error,
        'notify_process_stop': notify_process_stop
    })
    
    # Set flag running ke False
    ui_components['preprocessing_running'] = False
    
    return ui_components
