"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Operation handlers dengan implementasi yang disederhanakan dan check handler yang diperbaiki
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation menggunakan service reuse tanpa keyword extra"""
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
    """Execute check dengan implementasi yang diperbaiki menggunakan detector utils"""
    try:
        show_progress_safe(ui_components, 'check')
        
        # Import detector utils untuk check yang lebih akurat
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
        
        # Detect data location dan structure
        data_location = get_best_data_location()
        log_to_ui(ui_components, f"📁 Mengecek data di: {data_location}", 'info')
        
        # Detect raw dataset
        raw_info = detect_split_structure(data_location)
        if raw_info['status'] == 'success':
            raw_images = raw_info.get('total_images', 0)
            raw_labels = raw_info.get('total_labels', 0)
            available_splits = raw_info.get('available_splits', [])
            
            log_to_ui(ui_components, f"📊 Raw Dataset: {raw_images} gambar, {raw_labels} label", 'success')
            log_to_ui(ui_components, f"📂 Available splits: {', '.join(available_splits) if available_splits else 'flat structure'}", 'info')
        else:
            log_to_ui(ui_components, f"❌ Raw dataset tidak ditemukan: {raw_info.get('message', 'Unknown error')}", 'error')
        
        # Check augmented dataset
        try:
            aug_info = detect_split_structure(f"{data_location}/augmented")
            if aug_info['status'] == 'success' and aug_info.get('total_images', 0) > 0:
                aug_images = aug_info.get('total_images', 0)
                log_to_ui(ui_components, f"🔄 Augmented Dataset: {aug_images} file augmented", 'success')
            else:
                log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')
        except Exception:
            log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')
        
        # Check preprocessed dataset
        try:
            prep_info = detect_split_structure(f"{data_location}/preprocessed")
            if prep_info['status'] == 'success' and prep_info.get('total_images', 0) > 0:
                prep_images = prep_info.get('total_images', 0)
                log_to_ui(ui_components, f"📊 Preprocessed Dataset: {prep_images} file preprocessed", 'success')
            else:
                log_to_ui(ui_components, "📊 Preprocessed Dataset: Belum ada file preprocessed", 'info')
        except Exception:
            log_to_ui(ui_components, "📊 Preprocessed Dataset: Belum ada file preprocessed", 'info')
        
        # Summary
        ready_status = "✅ Siap" if raw_info['status'] == 'success' and raw_images > 0 else "❌ Tidak siap"
        log_to_ui(ui_components, f"🎯 Status augmentasi: {ready_status}", 'success' if ready_status.startswith('✅') else 'warning')
        
        complete_progress_safe(ui_components, "Check dataset selesai")
        
    except Exception as e:
        log_to_ui(ui_components, f"Check error: {str(e)}", 'error', "❌ ")
        error_progress_safe(ui_components, f"Check error: {str(e)}")

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
            total_generated = result.get('total_generated', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"Pipeline berhasil: {total_generated} file dalam {processing_time:.1f}s"
            
        elif operation == 'cleanup':
            total_deleted = result.get('total_deleted', 0)
            success_msg = f"Cleanup berhasil: {total_deleted} file dihapus"
        else:
            success_msg = f"{operation.title()} berhasil"
        
        log_to_ui(ui_components, success_msg, 'success', "✅ ")
        complete_progress_safe(ui_components, success_msg.split('\n')[0])
        
    else:
        error_msg = f"{operation.title()} gagal: {result.get('message', 'Unknown error')}"
        log_to_ui(ui_components, error_msg, 'error', "❌ ")
        error_progress_safe(ui_components, error_msg)

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'