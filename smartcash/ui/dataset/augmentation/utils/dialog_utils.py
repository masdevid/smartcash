"""
File: smartcash/ui/dataset/augmentation/utils/dialog_utils.py
Deskripsi: Dialog utilities untuk augmentation confirmations
"""

from typing import Dict, Any, Callable
from IPython.display import display

def show_cleanup_confirmation(ui_components: Dict[str, Any], on_confirm: Callable):
    """Show cleanup confirmation dengan shared dialog"""
    try:
        from smartcash.ui.components.dialogs import show_destructive_confirmation
        
        def cancel_handler(button):
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, "❌ Cleanup dibatalkan", 'info')
        
        dialog = show_destructive_confirmation(
            title="Konfirmasi Cleanup Dataset",
            message="Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
            item_name="file augmented",
            on_confirm=on_confirm,
            on_cancel=cancel_handler
        )
        
        _display_in_confirmation_area(ui_components, dialog)
        
    except ImportError:
        # Fallback jika dialog tidak tersedia
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, gunakan cleanup langsung", 'warning')
        on_confirm(None)

def show_reset_confirmation(ui_components: Dict[str, Any], on_confirm: Callable):
    """Show reset confirmation untuk config"""
    try:
        from smartcash.ui.components.dialogs import show_confirmation
        
        def cancel_handler(button):
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, "❌ Reset dibatalkan", 'info')
        
        dialog = show_confirmation(
            title="Konfirmasi Reset Konfigurasi",
            message="Apakah Anda yakin ingin reset konfigurasi ke default?\n\nSemua pengaturan saat ini akan hilang.",
            on_confirm=on_confirm,
            on_cancel=cancel_handler
        )
        
        _display_in_confirmation_area(ui_components, dialog)
        
    except ImportError:
        # Direct execute jika dialog tidak tersedia
        on_confirm(None)

def _display_in_confirmation_area(ui_components: Dict[str, Any], dialog):
    """Display dialog dalam confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output()
        with confirmation_area:
            display(dialog)
    else:
        # Fallback direct display
        display(dialog)