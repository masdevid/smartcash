"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Operation handlers dengan progress tracking dan button state management yang diperbaiki
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui, show_progress_safe, complete_progress_safe, error_progress_safe

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation dengan progress tracking dan button state management"""
    try:
        # Disable buttons dan show progress
        _disable_operation_buttons(ui_components)
        show_progress_safe(ui_components, 'augmentation')
        
        log_to_ui(ui_components, "🚀 Memulai pipeline augmentasi...", 'info')
        
        service = _create_service_safely(ui_components)
        if not service:
            log_to_ui(ui_components, "Gagal membuat augmentation service", 'error', "❌ ")
            return
        
        target_split = _get_target_split_safe(ui_components)
        log_to_ui(ui_components, f"📂 Target split: {target_split}", 'info')
        
        # Execute dengan progress callback yang terlihat
        result = service.run_full_augmentation_pipeline(
            target_split=target_split,
            progress_callback=_create_visible_progress_callback(ui_components)
        )
        
        _handle_service_result(ui_components, result, 'augmentation pipeline')
        
    except Exception as e:
        log_to_ui(ui_components, f"Pipeline error: {str(e)}", 'error', "❌ ")
    finally:
        # Re-enable buttons
        _enable_operation_buttons(ui_components)

def execute_check(ui_components: Dict[str, Any]):
    """Execute check dengan progress tracking"""
    try:
        _disable_operation_buttons(ui_components)
        show_progress_safe(ui_components, 'check')
        
        log_to_ui(ui_components, "🔍 Memulai pengecekan dataset...", 'info')
        
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
        
        # Progress update
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            tracker.update('overall', 20, "Mencari lokasi data")
        
        data_location = get_best_data_location()
        log_to_ui(ui_components, f"📁 Mengecek data di: {data_location}", 'info')
        
        # Check augmented dataset dengan progress
        if tracker and hasattr(tracker, 'update'):
            tracker.update('overall', 40, "Mengecek raw dataset")
        
        raw_info = detect_split_structure(data_location)
        if raw_info['status'] == 'success':
            raw_images = raw_info.get('total_images', 0)
            raw_labels = raw_info.get('total_labels', 0)
            available_splits = raw_info.get('available_splits', [])
            
            log_to_ui(ui_components, f"📊 Raw Dataset: {raw_images} gambar, {raw_labels} label", 'success')
            log_to_ui(ui_components, f"📂 Available splits: {', '.join(available_splits) if available_splits else 'flat structure'}", 'info')
        else:
            log_to_ui(ui_components, f"❌ Raw dataset tidak ditemukan: {raw_info.get('message', 'Unknown error')}", 'error')
        
        # Check augmented dataset dengan progress
        if tracker and hasattr(tracker, 'update'):
            tracker.update('overall', 70, "Mengecek augmented dataset")
        
        try:
            aug_info = detect_split_structure(f"{data_location}/augmented")
            if aug_info['status'] == 'success' and aug_info.get('total_images', 0) > 0:
                aug_images = aug_info.get('total_images', 0)
                log_to_ui(ui_components, f"🔄 Augmented Dataset: {aug_images} file augmented", 'success')
            else:
                log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')
        except Exception:
            log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')
        
        # Check preprocessed dataset dengan progress
        if tracker and hasattr(tracker, 'update'):
            tracker.update('overall', 90, "Mengecek preprocessed dataset")
        
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
    finally:
        _enable_operation_buttons(ui_components)

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

def _create_visible_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback yang terlihat dengan throttling"""
    import time
    last_update = {'time': 0, 'percentage': -1}
    
    def callback(step: str, current: int, total: int, message: str):
        current_time = time.time()
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        
        # Update setiap 0.3 detik atau perubahan signifikan
        if (current_time - last_update['time'] > 0.3 or 
            abs(percentage - last_update['percentage']) >= 5 or
            percentage in [0, 100]):
            
            last_update['time'] = current_time
            last_update['percentage'] = percentage
            
            # Log progress message
            log_to_ui(ui_components, f"📊 {message} ({percentage}%)", 'info')
            
            # Update tracker
            tracker = ui_components.get('tracker')
            if tracker and hasattr(tracker, 'update'):
                tracker.update(step, percentage, message)
    
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

def _disable_operation_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons saat proses berjalan"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = True
            # Update description jika bisa
            if hasattr(button, 'description'):
                original_desc = getattr(button, '_original_description', button.description)
                if not hasattr(button, '_original_description'):
                    button._original_description = button.description
                button.description = f"⏳ {original_desc}"

def _enable_operation_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons setelah proses selesai"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = False
            # Restore original description
            if hasattr(button, '_original_description'):
                button.description = button._original_description

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'