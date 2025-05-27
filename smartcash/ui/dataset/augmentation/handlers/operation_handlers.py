"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handler.py
Deskripsi: Handler untuk operations augmentation dan check dengan komunikasi ke service
"""

from typing import Dict, Any

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation dengan komunikasi ke augmentor service"""
    try:
        # Show progress untuk augmentation
        tracker = ui_components.get('tracker')
        tracker and hasattr(tracker, 'show') and tracker.show('augmentation')
        
        # Create service dari augmentor dengan UI components
        from smartcash.dataset.augmentor.service import create_service_from_ui
        service = create_service_from_ui(ui_components)
        
        # Execute pipeline
        result = service.run_full_augmentation_pipeline(
            target_split=_get_target_split(ui_components),
            progress_callback=_create_progress_callback(ui_components)
        )
        
        # Handle result
        _handle_operation_result(ui_components, result, 'augmentation')
        
    except Exception as e:
        _handle_operation_error(ui_components, f"Augmentation error: {str(e)}")

def execute_check(ui_components: Dict[str, Any]):
    """Execute check dataset dengan komunikasi ke service"""
    try:
        # Show progress untuk check
        tracker = ui_components.get('tracker')
        tracker and hasattr(tracker, 'show') and tracker.show('check')
        
        # Create service dan check status
        from smartcash.dataset.augmentor.service import create_service_from_ui
        service = create_service_from_ui(ui_components)
        status = service.get_augmentation_status()
        
        # Format status message
        raw_info = status.get('raw_dataset', {})
        aug_info = status.get('augmented_dataset', {})
        prep_info = status.get('preprocessed_dataset', {})
        
        status_lines = [
            f"📁 Raw: {'✅' if raw_info.get('exists') else '❌'} ({raw_info.get('total_images', 0)} img, {raw_info.get('total_labels', 0)} lbl)",
            f"🔄 Aug: {'✅' if aug_info.get('exists') else '❌'} ({aug_info.get('total_images', 0)} files)",
            f"📊 Prep: {'✅' if prep_info.get('exists') else '❌'} ({prep_info.get('total_files', 0)} files)",
            f"🎯 Ready: {'✅' if status.get('ready_for_augmentation') else '❌'}"
        ]
        
        result_message = "📊 Dataset Status:\n" + "\n".join(status_lines)
        _log_to_ui(ui_components, result_message, 'info')
        
        # Complete check
        tracker and hasattr(tracker, 'complete') and tracker.complete("Dataset check selesai")
        
    except Exception as e:
        _handle_operation_error(ui_components, f"Check error: {str(e)}")

def execute_cleanup(ui_components: Dict[str, Any]):
    """Execute cleanup dengan komunikasi ke service"""
    try:
        # Show progress untuk cleanup
        tracker = ui_components.get('tracker')
        tracker and hasattr(tracker, 'show') and tracker.show('cleanup')
        
        # Create service dan cleanup
        from smartcash.dataset.augmentor.service import create_service_from_ui
        service = create_service_from_ui(ui_components)
        
        result = service.cleanup_augmented_data(
            include_preprocessed=True,
            progress_callback=_create_progress_callback(ui_components)
        )
        
        # Handle result
        _handle_operation_result(ui_components, result, 'cleanup')
        
    except Exception as e:
        _handle_operation_error(ui_components, f"Cleanup error: {str(e)}")

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk service integration"""
    def callback(step: str, current: int, total: int, message: str):
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            percentage = min(100, int((current / max(1, total)) * 100))
            tracker.update(step, percentage, message)
    return callback

def _handle_operation_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle operation result dengan UI feedback"""
    tracker = ui_components.get('tracker')
    
    if result.get('status') == 'success':
        if operation == 'augmentation':
            success_msg = f"✅ Pipeline berhasil: {result.get('total_files', 0)} file dalam {result.get('processing_time', 0):.1f}s"
        elif operation == 'cleanup':
            success_msg = f"✅ Cleanup berhasil: {result.get('stats', {}).get('total_files_removed', 0)} file dihapus"
        else:
            success_msg = f"✅ {operation.title()} berhasil"
            
        _log_to_ui(ui_components, success_msg, 'success')
        tracker and hasattr(tracker, 'complete') and tracker.complete(success_msg)
    else:
        error_msg = f"❌ {operation.title()} gagal: {result.get('message', 'Unknown error')}"
        _log_to_ui(ui_components, error_msg, 'error')
        tracker and hasattr(tracker, 'error') and tracker.error(error_msg)

def _handle_operation_error(ui_components: Dict[str, Any], error_msg: str):
    """Handle operation error dengan UI feedback"""
    _log_to_ui(ui_components, error_msg, 'error')
    tracker = ui_components.get('tracker')
    tracker and hasattr(tracker, 'error') and tracker.error(error_msg)

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message ke UI log_output saja"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)

def _get_target_split(ui_components: Dict[str, Any]) -> str:
    """Get target split dari UI dengan default train"""
    target_split_widget = ui_components.get('target_split')
    return getattr(target_split_widget, 'value', 'train') if target_split_widget else 'train'