"""
File: smartcash/ui/dataset/preprocessing/utils/ui_observers.py
Deskripsi: Utilitas observer pattern untuk UI preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS

def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """
    Notifikasi observer bahwa proses telah dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("preprocessing", {
            'split': split,
            'display_info': display_info
        })

def notify_process_complete(ui_components: Dict[str, Any], result: Dict[str, Any], display_info: str) -> None:
    """
    Notifikasi observer bahwa proses telah selesai dengan sukses.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil dari proses
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.info(f"{ICONS['success']} Preprocessing {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("preprocessing", result)

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """
    Notifikasi observer bahwa proses mengalami error.
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.error(f"{ICONS['error']} Error pada preprocessing: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("preprocessing", error_message)

def notify_process_stop(ui_components: Dict[str, Any], display_info: str = "") -> None:
    """
    Notifikasi observer bahwa proses telah dihentikan oleh pengguna.
    
    Args:
        ui_components: Dictionary komponen UI
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.warning(f"{ICONS['stop']} Proses preprocessing dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("preprocessing", {
            'display_info': display_info
        })

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
        disable: True untuk menonaktifkan, False untuk mengaktifkan
    """
    # Daftar komponen yang perlu dinonaktifkan
    disable_components = [
        'split_selector', 'config_accordion', 'options_accordion',
        'reset_button', 'preprocess_button', 'save_button'
    ]
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components and hasattr(ui_components[component], 'disabled'):
            ui_components[component].disabled = disable
