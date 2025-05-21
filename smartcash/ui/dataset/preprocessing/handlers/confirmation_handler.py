"""
File: smartcash/ui/dataset/preprocessing/handlers/confirmation_handler.py
Deskripsi: Handler untuk dialog konfirmasi preprocessing dataset
"""

from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.preprocessing.handlers.executor import execute_preprocessing

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
    
    # Log informasi konfirmasi
    log_message(ui_components, "Menampilkan konfirmasi preprocessing", "info", "❓")
    
    # Pastikan area konfirmasi visible dan memiliki properti yang tepat
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'layout'):
        ui_components['confirmation_area'].layout.display = 'block'
        ui_components['confirmation_area'].layout.visibility = 'visible'
        ui_components['confirmation_area'].layout.height = 'auto'
        ui_components['confirmation_area'].layout.min_height = '150px'
        ui_components['confirmation_area'].layout.margin = '10px 0'
        ui_components['confirmation_area'].layout.border = '1px solid #ddd'
        ui_components['confirmation_area'].layout.padding = '10px'
    
    # Fungsi untuk menjalankan preprocessing dan membersihkan dialog
    def confirm_and_execute():
        log_message(ui_components, "Konfirmasi preprocessing diterima", "info", "✅")
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        # Jalankan preprocessing
        execute_preprocessing(ui_components, config)
    
    # Fungsi untuk membatalkan preprocessing - Menerima parameter b dari callback
    def cancel_preprocessing(b=None):
        log_message(ui_components, "Preprocessing dibatalkan", "info", "ℹ️")
        update_status_panel(ui_components, "info", "Preprocessing dibatalkan")
        
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False
    
    # Bersihkan area konfirmasi terlebih dahulu
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Buat dan tampilkan dialog konfirmasi
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title="Konfirmasi Preprocessing Dataset",
            message=message,
            on_confirm=confirm_and_execute,
            on_cancel=cancel_preprocessing
        )
        display(dialog) 