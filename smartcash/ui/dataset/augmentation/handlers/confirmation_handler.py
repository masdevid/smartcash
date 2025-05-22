"""
File: smartcash/ui/dataset/augmentation/handlers/confirmation_handler.py
Deskripsi: Handler konfirmasi untuk augmentasi dataset
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel, disable_buttons
from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar
from smartcash.ui.dataset.augmentation.handlers.augmentation_executor import run_augmentation, process_augmentation_result

def confirm_augmentation(ui_components: Dict[str, Any]) -> None:
    """
    Tampilkan dialog konfirmasi untuk augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    
    log_message(ui_components, "Menunggu konfirmasi augmentasi dataset", "info", "⏳")
    
    # Buat pesan konfirmasi
    aug_types = _get_augmentation_types(ui_components)
    split = _get_selected_split(ui_components)
    message = f"Anda akan menjalankan augmentasi {', '.join(aug_types)} pada dataset {split}. "
    message += "Proses ini akan menghasilkan gambar tambahan untuk meningkatkan variasi dataset. "
    message += "Lanjutkan?"
    
    # Reset status konfirmasi
    ui_components['confirmation_result'] = False
    
    # Callback untuk konfirmasi
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        update_status_panel(ui_components, "Memulai proses augmentasi dataset", "info")
        ui_components['confirmation_result'] = True
        log_message(ui_components, "Konfirmasi augmentasi diterima", "info", "✅")
        _execute_augmentation_after_confirm(ui_components)
    
    # Callback untuk batal
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        update_status_panel(ui_components, "Augmentasi dibatalkan oleh pengguna", "info")
        ui_components['confirmation_result'] = False
        log_message(ui_components, "Augmentasi dibatalkan oleh pengguna", "info", "❌")
        _enable_augmentation_button(ui_components)
    
    # Tampilkan dialog
    dialog = create_confirmation_dialog(
        title="Konfirmasi Augmentasi Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)
    
    update_status_panel(ui_components, "Silakan konfirmasi untuk melanjutkan augmentasi", "warning")

def _execute_augmentation_after_confirm(ui_components: Dict[str, Any]) -> None:
    """Eksekusi augmentasi setelah konfirmasi diterima."""
    reset_progress_bar(ui_components)
    
    # Log parameter
    params = _get_augmentation_params(ui_components)
    log_message(ui_components, "Parameter augmentasi:", "info", "ℹ️")
    for key, value in params.items():
        log_message(ui_components, f"- {key}: {value}", "debug", "🔹")
    
    # Nonaktifkan tombol selama augmentasi
    disable_buttons(ui_components, True)
    
    # Jalankan augmentasi
    result = run_augmentation(ui_components)
    process_augmentation_result(ui_components, result)

def _get_augmentation_types(ui_components: Dict[str, Any]) -> list:
    """Dapatkan jenis augmentasi yang dipilih."""
    aug_types = ui_components.get('augmentation_types', {})
    if hasattr(aug_types, 'value'):
        return list(aug_types.value) if aug_types.value else ['combined']
    return ['combined']

def _get_selected_split(ui_components: Dict[str, Any]) -> str:
    """Dapatkan split yang dipilih."""
    split_selector = ui_components.get('split_selector', {})
    if hasattr(split_selector, 'value'):
        return split_selector.value
    return 'train'

def _get_augmentation_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Dapatkan parameter augmentasi dari UI."""
    return {
        'split': _get_selected_split(ui_components),
        'augmentation_types': _get_augmentation_types(ui_components),
        'num_variations': getattr(ui_components.get('num_variations', {}), 'value', 2),
        'output_prefix': getattr(ui_components.get('output_prefix', {}), 'value', 'aug'),
        'target_count': getattr(ui_components.get('target_count', {}), 'value', 1000)
    }

def _enable_augmentation_button(ui_components: Dict[str, Any]) -> None:
    """Aktifkan tombol augmentasi."""
    if 'augment_button' in ui_components:
        ui_components['augment_button'].disabled = False