"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Fixed handlers dengan proper service integration dan parameter alignment
"""

from typing import Dict, Any

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation dengan aligned parameters dan suppressed logging"""
    try:
        # Show progress untuk augmentation
        _show_progress_safe(ui_components, 'augmentation')
        
        # Validate UI parameters sebelum execution
        if not _validate_ui_parameters(ui_components):
            _handle_operation_error(ui_components, "❌ Parameter UI tidak valid - periksa konfigurasi")
            return
        
        # Create service dengan aligned config
        service = _create_service_safely(ui_components)
        if not service:
            _handle_operation_error(ui_components, "❌ Gagal membuat augmentation service")
            return
        
        # Get target split dari UI
        target_split = _get_target_split_safe(ui_components)
        
        # Execute pipeline dengan progress callback
        result = service.run_full_augmentation_pipeline(
            target_split=target_split,
            progress_callback=_create_progress_callback(ui_components)
        )
        
        # Handle result dengan detailed feedback
        _handle_operation_result(ui_components, result, 'augmentation')
        
    except Exception as e:
        _handle_operation_error(ui_components, f"Augmentation error: {str(e)}")

def execute_check(ui_components: Dict[str, Any]):
    """Execute check dengan comprehensive status reporting"""
    try:
        # Show progress untuk check
        _show_progress_safe(ui_components, 'check')
        
        # Create service dan check status
        service = _create_service_safely(ui_components)
        if not service:
            _handle_operation_error(ui_components, "❌ Gagal membuat service untuk check")
            return
            
        status = service.get_augmentation_status()
        
        # Format comprehensive status message
        status_message = _format_comprehensive_status(status)
        _log_to_ui_only(ui_components, status_message, 'info')
        
        # Complete check dengan summary
        total_files = (status.get('raw_dataset', {}).get('total_images', 0) + 
                      status.get('augmented_dataset', {}).get('total_images', 0) + 
                      status.get('preprocessed_dataset', {}).get('total_files', 0))
        
        _complete_progress_safe(ui_components, f"Check selesai: {total_files} total file terdeteksi")
        
    except Exception as e:
        _handle_operation_error(ui_components, f"Check error: {str(e)}")

def execute_cleanup(ui_components: Dict[str, Any]):
    """Execute cleanup dengan comprehensive progress tracking"""
    try:
        # Show progress untuk cleanup
        _show_progress_safe(ui_components, 'cleanup')
        
        # Create service dan cleanup
        service = _create_service_safely(ui_components)
        if not service:
            _handle_operation_error(ui_components, "❌ Gagal membuat service untuk cleanup")
            return
        
        result = service.cleanup_augmented_data(
            include_preprocessed=True,
            progress_callback=_create_progress_callback(ui_components)
        )
        
        # Handle result dengan detailed cleanup stats
        _handle_operation_result(ui_components, result, 'cleanup')
        
    except Exception as e:
        _handle_operation_error(ui_components, f"Cleanup error: {str(e)}")

def _create_service_safely(ui_components: Dict[str, Any]):
    """Create augmentation service dengan comprehensive error handling"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        return create_service_from_ui(ui_components)
    except ImportError as e:
        _log_to_ui_only(ui_components, f"❌ Service import error: {str(e)}", 'error')
        return None
    except Exception as e:
        _log_to_ui_only(ui_components, f"❌ Service creation error: {str(e)}", 'error')
        return None

def _validate_ui_parameters(ui_components: Dict[str, Any]) -> bool:
    """Validate UI parameters sebelum operation"""
    try:
        from smartcash.dataset.augmentor.config import validate_ui_parameters
        return validate_ui_parameters(ui_components)
    except Exception:
        return True  # Fallback allow operation

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback dengan throttling"""
    import time
    last_update = {'time': 0}
    
    def callback(step: str, current: int, total: int, message: str):
        # Throttle updates untuk prevent flooding
        current_time = time.time()
        if current_time - last_update['time'] < 0.5 and current != total:  # 500ms throttle
            return
        
        last_update['time'] = current_time
        
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            percentage = min(100, int((current / max(1, total)) * 100))
            tracker.update(step, percentage, message)
    
    return callback

def _handle_operation_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle operation result dengan comprehensive feedback"""
    tracker = ui_components.get('tracker')
    
    if result.get('status') == 'success':
        # Success messages dengan detailed stats
        if operation == 'augmentation':
            total_files = result.get('total_files', 0)
            processing_time = result.get('processing_time', 0)
            success_msg = f"✅ Pipeline berhasil: {total_files} file dalam {processing_time:.1f}s"
            
            # Additional stats dari steps
            steps = result.get('steps', {})
            aug_stats = steps.get('augmentation', {})
            norm_stats = steps.get('normalization', {})
            
            if aug_stats or norm_stats:
                stats_lines = []
                if aug_stats.get('total_generated'):
                    stats_lines.append(f"   🔄 Augmented: {aug_stats['total_generated']} file")
                if norm_stats.get('total_normalized'):
                    stats_lines.append(f"   📊 Normalized: {norm_stats['total_normalized']} file")
                if stats_lines:
                    success_msg += "\n" + "\n".join(stats_lines)
            
        elif operation == 'cleanup':
            stats = result.get('stats', {})
            files_removed = stats.get('total_files_removed', 0)
            folders_cleaned = len(stats.get('folders_cleaned', []))
            success_msg = f"✅ Cleanup berhasil: {files_removed} file dari {folders_cleaned} folder"
            
        else:
            success_msg = f"✅ {operation.title()} berhasil"
        
        _log_to_ui_only(ui_components, success_msg, 'success')
        _complete_progress_safe(ui_components, success_msg.split('\n')[0])  # First line only
        
    else:
        error_msg = f"❌ {operation.title()} gagal: {result.get('message', 'Unknown error')}"
        _log_to_ui_only(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)

def _handle_operation_error(ui_components: Dict[str, Any], error_msg: str):
    """Handle operation error dengan comprehensive feedback"""
    _log_to_ui_only(ui_components, error_msg, 'error')
    _error_progress_safe(ui_components, error_msg)

def _format_comprehensive_status(status: Dict[str, Any]) -> str:
    """Format comprehensive status message"""
    raw_info = status.get('raw_dataset', {})
    aug_info = status.get('augmented_dataset', {})
    prep_info = status.get('preprocessed_dataset', {})
    
    status_lines = [
        "📊 Dataset Status Comprehensive:",
        f"📁 Raw Dataset: {'✅' if raw_info.get('exists') else '❌'} "
        f"({raw_info.get('total_images', 0)} img, {raw_info.get('total_labels', 0)} lbl)",
        f"🔄 Augmented: {'✅' if aug_info.get('exists') else '❌'} "
        f"({aug_info.get('total_images', 0)} files)",
        f"📊 Preprocessed: {'✅' if prep_info.get('exists') else '❌'} "
        f"({prep_info.get('total_files', 0)} files)",
        f"🎯 Ready for Augmentation: {'✅' if status.get('ready_for_augmentation') else '❌'}"
    ]
    
    # Add error info jika ada
    if 'error' in status:
        status_lines.append(f"⚠️ Error: {status['error']}")
    
    return "\n".join(status_lines)

def _log_to_ui_only(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message hanya ke UI output, tidak ke console"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe fallback"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'

# Progress management dengan safe operations
def _show_progress_safe(ui_components: Dict[str, Any], operation: str):
    """Show progress dengan safe error handling"""
    try:
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'show'):
            tracker.show(operation)
        elif 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
    except Exception:
        pass

def _complete_progress_safe(ui_components: Dict[str, Any], message: str):
    """Complete progress dengan safe error handling"""
    try:
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'complete'):
            tracker.complete(message)
        elif 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
    except Exception:
        pass

def _error_progress_safe(ui_components: Dict[str, Any], message: str):
    """Error progress dengan safe error handling"""
    try:
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'error'):
            tracker.error(message)
        elif 'error_operation' in ui_components:
            ui_components['error_operation'](message)
    except Exception:
        pass