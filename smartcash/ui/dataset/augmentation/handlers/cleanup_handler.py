"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Cleanup handler dengan silent handling untuk empty dataset
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe

def show_cleanup_confirmation(ui_components: Dict[str, Any]):
    """Show cleanup confirmation dengan empty dataset check"""
    try:
        # Check dulu apakah ada data yang bisa dihapus
        has_data = _check_cleanup_data_exists(ui_components)
        
        if not has_data:
            log_to_ui(ui_components, "💡 Tidak ada file augmented untuk dihapus", 'info', "ℹ️ ")
            return
        
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        
        def confirm_cleanup(button):
            _execute_cleanup_with_progress(ui_components)
        
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

def _check_cleanup_data_exists(ui_components: Dict[str, Any]) -> bool:
    """Check apakah ada data augmented yang bisa dihapus"""
    try:
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
        
        data_location = get_best_data_location()
        
        # Check augmented data
        try:
            aug_info = detect_split_structure(f"{data_location}/augmented")
            if aug_info['status'] == 'success' and aug_info.get('total_images', 0) > 0:
                return True
        except Exception:
            pass
        
        # Check preprocessed data
        try:
            prep_info = detect_split_structure(f"{data_location}/preprocessed")
            if prep_info['status'] == 'success' and prep_info.get('total_images', 0) > 0:
                return True
        except Exception:
            pass
        
        return False
        
    except Exception:
        return False  # Assume no data jika error

def _execute_cleanup_with_progress(ui_components: Dict[str, Any]):
    """Execute cleanup dengan progress tracking dan button state"""
    try:
        # Disable buttons
        _disable_cleanup_buttons(ui_components)
        show_progress_safe(ui_components, 'cleanup')
        
        log_to_ui(ui_components, "🧹 Memulai cleanup dataset...", 'info')
        
        from smartcash.dataset.augmentor.service import create_service_from_ui
        
        service = create_service_from_ui(ui_components)
        
        # Progress update
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            tracker.update('overall', 30, 100, "Mencari file augmented")
        
        result = service.cleanup_augmented_data(include_preprocessed=True)
        
        # Progress update
        if tracker and hasattr(tracker, 'update'):
            tracker.update('overall', 80, 100, "Menyelesaikan cleanup")
        
        if result.get('status') == 'success':
            total_deleted = result.get('total_deleted', 0)
            if total_deleted > 0:
                log_to_ui(ui_components, f"Cleanup berhasil: {total_deleted} file dihapus", 'success', "✅ ")
                complete_progress_safe(ui_components, f"Cleanup selesai: {total_deleted} files")
            else:
                log_to_ui(ui_components, "💡 Cleanup selesai: tidak ada file untuk dihapus", 'info', "ℹ️ ")
                complete_progress_safe(ui_components, "Cleanup selesai: no files")
        elif result.get('status') == 'empty':
            log_to_ui(ui_components, "💡 Cleanup selesai: tidak ada file untuk dihapus", 'info', "ℹ️ ")
            complete_progress_safe(ui_components, "Cleanup selesai: no files")
        else:
            log_to_ui(ui_components, f"Cleanup warning: {result.get('message', 'Unknown issue')}", 'warning', "⚠️ ")
            complete_progress_safe(ui_components, "Cleanup completed with warnings")
        
    except Exception as e:
        error_msg = f"Cleanup error: {str(e)}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)
    finally:
        # Re-enable buttons
        _enable_cleanup_buttons(ui_components)

def _disable_cleanup_buttons(ui_components: Dict[str, Any]):
    """Disable cleanup-related buttons"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = True

def _enable_cleanup_buttons(ui_components: Dict[str, Any]):
    """Enable cleanup-related buttons"""  
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = False