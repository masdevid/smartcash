"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Operation handlers dengan unified logging dan service integration
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation menggunakan service reuse"""
    try:
        show_progress_safe(ui_components, 'augmentation')
        
        service = _create_service_safely(ui_components)
        if not service:
            log_to_ui(ui_components, "Gagal membuat augmentation service", 'error', "❌ ")
            return
        
        target_split = _get_target_split_safe(ui_components)
        result = service.run_full_augmentation_pipeline(
            target_split=target_split,
            progress_callback=_create_progress_callback(ui_components)
        )
        
        _handle_service_result(ui_components, result, 'augmentation pipeline')
        
    except Exception as e:
        log_to_ui(ui_components, f"Pipeline error: {str(e)}", 'error', "❌ ")

def execute_check(ui_components: Dict[str, Any]):
    """Execute check menggunakan service status"""
    try:
        show_progress_safe(ui_components, 'check')
        
        service = _create_service_safely(ui_components)
        if not service:
            log_to_ui(ui_components, "Gagal membuat service untuk check", 'error', "❌ ")
            return
            
        status = service.get_augmentation_status()
        status_message = _format_service_status(status)
        log_to_ui(ui_components, status_message, 'info')
        
        complete_progress_safe(ui_components, "Check selesai: service ready")
        
    except Exception as e:
        log_to_ui(ui_components, f"Check error: {str(e)}", 'error', "❌ ")

def _create_service_safely(ui_components: Dict[str, Any]):
    """Create augmentation service dengan UI config integration"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        return create_service_from_ui(ui_components)
        
    except ImportError as e:
        log_to_ui(ui_components, f"Service import error: {str(e)}", 'error', "❌ ")
        return None
    except Exception as e:
        log_to_ui(ui_components, f"Service creation error: {str(e)}", 'error', "❌ ")
        return None

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback dengan service integration"""
    import time
    last_update = {'time': 0}
    
    def callback(step: str, current: int, total: int, message: str):
        current_time = time.time()
        if current_time - last_update['time'] < 0.5 and current != total:
            return
        
        last_update['time'] = current_time
        
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            percentage = min(100, int((current / max(1, total)) * 100))
            tracker.update(step, percentage, f"🎯 {message}")
    
    return callback

def _handle_service_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle service result dengan comprehensive feedback"""
    if result.get('status') == 'success':
        if operation == 'augmentation pipeline':
            total_generated = result.get('total_files', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"Pipeline berhasil: {total_generated} file dalam {processing_time:.1f}s"
            
            steps = result.get('steps', {})
            if steps:
                success_msg += f"\n📊 Steps completed: {len(steps)}"
            
        elif operation == 'cleanup':
            total_deleted = result.get('total_files_removed', 0)
            success_msg = f"Cleanup berhasil: {total_deleted} file dihapus"
        else:
            success_msg = f"{operation.title()} berhasil"
        
        log_to_ui(ui_components, success_msg, 'success', "✅ ")
        complete_progress_safe(ui_components, success_msg.split('\n')[0])
        
    else:
        error_msg = f"{operation.title()} gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)

def _format_service_status(status: Dict[str, Any]) -> str:
    """Format service status message"""
    status_lines = [
        "🎯 Service Status:",
        f"📂 Raw Dataset: {'✅' if status.get('raw_dataset', {}).get('exists') else '❌'}",
        f"🔄 Augmented: {'✅' if status.get('augmented_dataset', {}).get('exists') else '❌'}",
        f"📊 Preprocessed: {'✅' if status.get('preprocessed_dataset', {}).get('exists') else '❌'}",
        f"🎯 Ready: {'✅' if status.get('ready_for_augmentation') else '❌'}"
    ]
    
    return "\n".join(status_lines)

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'