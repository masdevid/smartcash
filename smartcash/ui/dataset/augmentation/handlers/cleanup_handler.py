"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Cleanup handler dengan progress tracking dan error handling
"""

from typing import Dict, Any

def execute_cleanup_with_progress(ui_components: Dict[str, Any]):
    """Execute cleanup dengan progress tracking dan safe operations"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe
        
        # Show progress untuk cleanup
        show_progress_safe(ui_components, 'cleanup')
        log_to_ui(ui_components, "🧹 Memulai cleanup dataset...", 'info')
        
        # Create service
        from smartcash.dataset.augmentor.service import create_service_from_ui
        service = create_service_from_ui(ui_components)
        
        # Execute cleanup
        result = service.cleanup_augmented_data(include_preprocessed=True)
        
        # Handle result
        if result.get('status') == 'success':
            total_deleted = result.get('total_deleted', 0)
            if total_deleted > 0:
                message = f"✅ Cleanup berhasil: {total_deleted} file dihapus"
                log_to_ui(ui_components, message, 'success')
                complete_progress_safe(ui_components, message)
            else:
                message = "💡 Cleanup selesai: tidak ada file untuk dihapus"
                log_to_ui(ui_components, message, 'info')
                complete_progress_safe(ui_components, message)
        else:
            error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
            log_to_ui(ui_components, error_msg, 'error')
            error_progress_safe(ui_components, error_msg)
            
    except Exception as e:
        error_msg = f"❌ Cleanup error: {str(e)}"
        log_to_ui(ui_components, error_msg, 'error')
        error_progress_safe(ui_components, error_msg)