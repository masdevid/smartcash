"""
File: smartcash/ui/dataset/augmentation/handlers/confirmation_handler.py
Deskripsi: Handler untuk konfirmasi augmentasi dengan logger bridge (SRP)
"""

from typing import Dict, Any, Callable
from IPython.display import display

def show_augmentation_confirmation(ui_components: Dict[str, Any], params: Dict[str, Any], 
                                 ui_logger, on_confirm_callback: Callable) -> None:
    """
    Tampilkan dialog konfirmasi untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi yang sudah divalidasi
        ui_logger: UI Logger bridge
        on_confirm_callback: Callback yang dipanggil saat konfirmasi diterima
    """
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    
    # Ensure confirmation area exists
    _ensure_confirmation_area(ui_components)
    
    # Build confirmation message
    message = _build_confirmation_message(params)
    
    # Setup callbacks
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        ui_logger.info("✅ Konfirmasi augmentasi diterima")
        on_confirm_callback()
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        ui_logger.info("❌ Augmentasi dibatalkan oleh pengguna")
    
    # Create and display dialog
    dialog = create_confirmation_dialog(
        title="Konfirmasi Augmentasi Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)
    
    ui_logger.info("⏳ Menunggu konfirmasi pengguna")

def _build_confirmation_message(params: Dict[str, Any]) -> str:
    """Build confirmation message dari parameter."""
    aug_types = params.get('types', ['combined'])
    split = params.get('split', 'train')
    target_count = params.get('target_count', 500)
    num_variations = params.get('num_variations', 2)
    
    message = f"Augmentasi {', '.join(aug_types)} pada dataset {split}.\n"
    message += f"Target: {target_count} instance per kelas, {num_variations} variasi per gambar.\n"
    
    # Tambahkan info storage
    if params.get('uses_symlink', False):
        message += "Hasil akan otomatis tersimpan ke Google Drive via symlink.\n"
    else:
        message += "Hasil akan tersimpan di storage lokal.\n"
    
    message += "\nLanjutkan proses augmentasi?"
    
    return message

def _ensure_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Pastikan confirmation area tersedia."""
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()
        
        # Tambahkan ke UI jika memungkinkan
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                children = list(ui_components['ui'].children)
                children.append(ui_components['confirmation_area'])
                ui_components['ui'].children = tuple(children)
            except Exception:
                pass  # Gagal menambahkan tidak masalah