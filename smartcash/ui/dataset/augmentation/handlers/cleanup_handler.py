"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Cleanup handler dengan unified logging dan simple fallback
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe

def show_cleanup_confirmation(ui_components: Dict[str, Any]):
    """Show cleanup confirmation dengan simple fallback"""
    try:
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        
        def confirm_cleanup(button):
            _execute_simple_cleanup(ui_components)
        
        def cancel_cleanup(button):
            log_to_ui(ui_components, "Cleanup dibatalkan", 'info', "❌ ")
        
        dialog = create_confirmation_dialog(
            title="Konfirmasi Cleanup Dataset",
            message="Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
            on_confirm=confirm_cleanup,
            on_cancel=cancel_cleanup,
            danger_mode=True
        )
        
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output()
            with confirmation_area:
                display(dialog)
        else:
            display(dialog)
            
    except ImportError:
        log_to_ui(ui_components, "Cannot show confirmation dialog", 'error', "❌ ")

def _execute_simple_cleanup(ui_components: Dict[str, Any]):
    """Execute simple cleanup dengan fallback operations"""
    try:
        show_progress_safe(ui_components, 'cleanup')
        
        from smartcash.dataset.augmentor.service import create_service_from_ui
        
        service = create_service_from_ui(ui_components)
        result = service.cleanup_augmented_data(include_preprocessed=True)
        
        if result.get('status') == 'success':
            total_deleted = result.get('total_deleted', 0)
            log_to_ui(ui_components, f"Cleanup berhasil: {total_deleted} file dihapus", 'success', "✅ ")
            complete_progress_safe(ui_components, f"Cleanup selesai: {total_deleted} files")
        else:
            log_to_ui(ui_components, f"Cleanup gagal: {result.get('message', 'Unknown error')}", 'error', "❌ ")
            error_progress_safe(ui_components, "Cleanup failed")
        
    except Exception as e:
        error_msg = f"Cleanup error: {str(e)}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)