"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler untuk cleanup confirmation dan execution
"""

from typing import Dict, Any
from IPython.display import display

def show_cleanup_confirmation(ui_components: Dict[str, Any]):
    """Show cleanup confirmation dialog"""
    try:
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        
        def confirm_cleanup(button):
            from smartcash.ui.dataset.augmentation.handlers.operation_handlers import execute_cleanup
            execute_cleanup(ui_components)
        
        def cancel_cleanup(button):
            _log_to_ui(ui_components, "❌ Cleanup dibatalkan", 'info')
        
        dialog = create_confirmation_dialog(
            title="Konfirmasi Cleanup Dataset",
            message="Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
            on_confirm=confirm_cleanup,
            on_cancel=cancel_cleanup,
            danger_mode=True
        )
        
        # Show dalam confirmation area
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output()
            with confirmation_area:
                display(dialog)
        else:
            display(dialog)
            
    except ImportError:
        _log_to_ui(ui_components, "❌ Cannot show confirmation dialog", 'error')

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message ke UI"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)