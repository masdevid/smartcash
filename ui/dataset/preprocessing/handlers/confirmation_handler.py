"""
File: smartcash/ui/dataset/preprocessing/handlers/confirmation_handler.py
Deskripsi: Handler konfirmasi untuk preprocessing dataset
"""

from typing import Dict, Any, Callable, Optional, List
from IPython.display import display, clear_output
import ipywidgets as widgets
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel, toggle_input_controls
from smartcash.ui.dataset.preprocessing.utils.progress_manager import reset_progress_bar
from smartcash.ui.dataset.preprocessing.utils.notification_manager import notify_status, PREPROCESSING_LOGGER_NAMESPACE
from smartcash.ui.dataset.preprocessing.utils.ui_helpers import ensure_output_area

def confirm_preprocessing(ui_components: Dict[str, Any], split: Optional[str] = None) -> None:
    """
    Tampilkan dialog konfirmasi untuk preprocessing dataset.
    Menggunakan callback pattern yang kompatibel dengan Colab.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split yang akan diproses (None untuk semua split)
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Import modul yang diperlukan
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    
    # Log message sebelum konfirmasi
    log_message(ui_components, "Menunggu konfirmasi preprocessing dataset", "info", "⏳")
    
    # Dapatkan informasi split
    split_text = split if split else "semua split"
    
    # Dapatkan informasi konfigurasi
    config = ui_components.get('config', {}).get('preprocessing', {})
    img_size = config.get('img_size', 640)
    normalizing = "dengan" if config.get('normalization', {}).get('enabled', True) else "tanpa"
    aspect_ratio = "mempertahankan" if config.get('normalization', {}).get('preserve_aspect_ratio', True) else "tidak mempertahankan"
    
    # Buat pesan konfirmasi
    message = f"Anda akan melakukan preprocessing dataset untuk {split_text} "
    message += f"dengan resolusi {img_size}px, {normalizing} normalisasi, dan {aspect_ratio} aspek rasio. "
    
    if config.get('validate', {}).get('enabled', True):
        message += "Validasi dataset akan dilakukan. "
        if config.get('validate', {}).get('fix_issues', True):
            message += "Isu-isu kecil akan otomatis diperbaiki. "
        if config.get('validate', {}).get('move_invalid', True):
            invalid_dir = config.get('validate', {}).get('invalid_dir', "invalid")
            message += f"File tidak valid akan dipindahkan ke direktori '{invalid_dir}'. "
    
    # Pastikan area konfirmasi tersedia
    ui_components = ensure_output_area(ui_components, 'confirmation_area', 'main_output')
    
    # Reset status konfirmasi
    ui_components['confirmation_result'] = False
    
    # Callback untuk tombol konfirmasi
    def on_confirm(b):
        # Bersihkan area konfirmasi
        if hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Update status panel
        update_status_panel(ui_components, "started", "Memulai proses preprocessing dataset")
        
        # Set hasil konfirmasi
        ui_components['confirmation_result'] = True
        
        # Log konfirmasi
        log_message(ui_components, "Konfirmasi preprocessing diterima", "info", "✅")
        
        # Lanjutkan dengan proses preprocessing
        _execute_preprocessing_after_confirm(ui_components, split)
    
    # Callback untuk tombol batal
    def on_cancel(b):
        # Bersihkan area konfirmasi
        if hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Update status panel
        update_status_panel(ui_components, "idle", "Preprocessing dibatalkan oleh pengguna")
        
        # Set hasil konfirmasi
        ui_components['confirmation_result'] = False
        
        # Log pembatalan
        log_message(ui_components, "Preprocessing dibatalkan oleh pengguna", "info", "❌")
        
        # Aktifkan kembali tombol
        toggle_input_controls(ui_components, False)
    
    # Gunakan component dialog konfirmasi
    dialog = create_confirmation_dialog(
        title="Konfirmasi Preprocessing Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_label="Mulai Preprocessing",
        cancel_label="Batal"
    )
    
    # Tampilkan dialog di area konfirmasi
    if hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
        with ui_components['confirmation_area']:
            display(dialog)
    
    # Update status panel
    update_status_panel(ui_components, "warning", "Silakan konfirmasi untuk melanjutkan preprocessing dataset")
    
    # Notifikasi status
    notify_status(ui_components, "confirm", f"Menunggu konfirmasi untuk preprocessing {split_text}")

def _execute_preprocessing_after_confirm(ui_components: Dict[str, Any], split: Optional[str] = None) -> None:
    """
    Eksekusi preprocessing setelah konfirmasi diterima.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split yang akan diproses (None untuk semua split)
    """
    try:
        # Reset progress bar setelah konfirmasi
        reset_progress_bar(ui_components)
        
        # Set flag running ke True
        ui_components['preprocessing_running'] = True
        
        # Map split ke informasi yang lebih deskriptif
        split_map = {
            'train': 'Train Only',
            'valid': 'Validation Only',
            'test': 'Test Only',
            None: 'All Splits'
        }
        split_info = split_map.get(split, "All Splits")
        
        # Log pada UI
        log_message(ui_components, f"Memulai preprocessing untuk {split_info}", "info", "🚀")
        
        # Import fungsi execute_preprocessing dari button_handler
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import execute_preprocessing
        
        # Eksekusi preprocessing
        execute_preprocessing(ui_components, split, split_info)
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat memulai preprocessing: {str(e)}", "error", "❌")
        
        # Update status panel
        update_status_panel(ui_components, "error", f"Error: {str(e)}")
        
        # Aktifkan kembali tombol
        toggle_input_controls(ui_components, False)
        
        # Set flag running ke False
        ui_components['preprocessing_running'] = False

def setup_confirmation_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk konfirmasi preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Pastikan area konfirmasi tersedia
    ui_components = ensure_output_area(ui_components, 'confirmation_area', 'main_output')
    
    # Tambahkan fungsi ke ui_components
    ui_components['confirm_preprocessing'] = confirm_preprocessing
    
    # Log setup berhasil
    log_message(ui_components, "Confirmation handler berhasil disetup", "debug", "✅")
    
    return ui_components 