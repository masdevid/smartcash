"""
File: smartcash/ui/dataset/download/handlers/reset_handler.py
Deskripsi: Handler untuk reset UI dan state pada modul download
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.ui_logger import log_to_ui
from smartcash.ui.dataset.download.utils.logger_helper import log_message

def handle_reset_button_click(ui_components: Dict[str, Any], b: Any = None) -> None:
    """
    Handler untuk tombol reset pada UI download.
    
    Args:
        ui_components: Dictionary komponen UI
        b: Button widget (opsional)
    """
    # Reset log output saat tombol diklik
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    try:
        # Reset UI components menggunakan konfigurasi default dari config manager
        reset_download_ui(ui_components)
        
        # Log reset berhasil
        log_message(ui_components, "UI download berhasil direset ke konfigurasi default", "info", "🔄")
        
        # Update status panel jika tersedia
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel'](ui_components, "info", "Konfigurasi download dataset")
        
    except Exception as e:
        # Tampilkan error
        error_msg = f"Error saat reset UI: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")

def reset_download_ui(ui_components: Dict[str, Any]) -> None:
    """
    Reset semua komponen UI download ke nilai default dari SimpleConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Ambil konfigurasi default dari config manager
        config_manager = get_config_manager()
        default_config = config_manager.get_module_config('dataset')
        
        # Default values jika config tidak ada
        default_values = {
            'url': '',
            'type': 'currency',
            'save_path': 'data/raw',
            'auto_extract': True,
            'validate': True
        }
        
        # Gunakan nilai dari config jika tersedia, jika tidak gunakan default
        download_config = default_config.get('download', {})
        
        # Reset input fields berdasarkan konfigurasi
        if 'url_input' in ui_components:
            ui_components['url_input'].value = download_config.get('url', default_values['url'])
        
        if 'dataset_type' in ui_components:
            ui_components['dataset_type'].value = download_config.get('type', default_values['type'])
        
        if 'save_path' in ui_components:
            ui_components['save_path'].value = download_config.get('save_path', default_values['save_path'])
        
        if 'auto_extract' in ui_components:
            ui_components['auto_extract'].value = download_config.get('auto_extract', default_values['auto_extract'])
        
        if 'validate_dataset' in ui_components:
            ui_components['validate_dataset'].value = download_config.get('validate', default_values['validate'])
            
        log_message(ui_components, f"Reset download UI dengan konfigurasi: {download_config}", "debug", "🔄")
        
    except Exception as e:
        log_message(ui_components, f"Tidak bisa memuat konfigurasi default: {str(e)}. Menggunakan fallback.", "warning", "⚠️")
        # Fallback ke reset original
        _reset_download_ui_fallback(ui_components)
    
    # Reset progress tracking
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar']()
    else:
        # Fallback ke reset progress bar manual
        _reset_progress_bar(ui_components)
    
    # Reset summary container
    if 'summary_container' in ui_components:
        ui_components['summary_container'].clear_output()
        ui_components['summary_container'].layout.display = 'none'
    
    # Aktifkan tombol-tombol
    _enable_buttons(ui_components)

def _reset_download_ui_fallback(ui_components: Dict[str, Any]) -> None:
    """
    Metode fallback untuk reset UI jika SimpleConfigManager gagal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset input fields ke nilai default hard-coded
    if 'url_input' in ui_components:
        ui_components['url_input'].value = ''
    
    if 'dataset_type' in ui_components:
        ui_components['dataset_type'].value = 'currency'
    
    if 'save_path' in ui_components:
        ui_components['save_path'].value = 'data/raw'
    
    if 'auto_extract' in ui_components:
        ui_components['auto_extract'].value = True
    
    if 'validate_dataset' in ui_components:
        ui_components['validate_dataset'].value = True

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke nilai awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Reset labels
    for label_key in ['overall_label', 'step_label']:
        if label_key in ui_components:
            ui_components[label_key].value = ""
            ui_components[label_key].layout.visibility = 'hidden'
    
    # Reset current progress
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = 0
        ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Reset progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.visibility = 'hidden'

def _enable_buttons(ui_components: Dict[str, Any]) -> None:
    """
    Aktifkan semua tombol UI.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Daftar tombol yang perlu diaktifkan
    button_keys = ['download_button', 'check_button', 'reset_button']
    
    # Set status enabled untuk semua tombol
    for key in button_keys:
        if key in ui_components:
            ui_components[key].disabled = False
            if hasattr(ui_components[key], 'layout'):
                ui_components[key].layout.display = 'block'
