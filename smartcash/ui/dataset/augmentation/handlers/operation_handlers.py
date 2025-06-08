"""
File: smartcash/ui/dataset/augmentation/handlers/operation_handlers.py
Deskripsi: Operation handlers dengan new progress tracker API dan granular tracking
"""

from typing import Dict, Any

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation dengan new progress tracker API"""
    try:
        # Disable buttons dan show progress
        _disable_operation_buttons(ui_components)
        
        # Show progress dengan steps dari config
        _show_progress_for_operation(ui_components, 'augmentation')
        
        _log_to_ui(ui_components, "🚀 Memulai pipeline augmentasi...", 'info')
        
        service = _create_service_safely(ui_components)
        if not service:
            _log_to_ui(ui_components, "❌ Gagal membuat augmentation service", 'error')
            return
        
        target_split = _get_target_split_safe(ui_components)
        _log_to_ui(ui_components, f"📂 Target split: {target_split}", 'info')
        
        # Execute dengan new progress callback
        result = service.run_full_augmentation_pipeline(
            target_split=target_split,
            progress_callback=_create_new_progress_callback(ui_components)
        )
        
        _handle_service_result(ui_components, result, 'augmentation pipeline')
        
    except Exception as e:
        error_msg = f"❌ Pipeline error: {str(e)}"
        _log_to_ui(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)
    finally:
        _enable_operation_buttons(ui_components)

def execute_check(ui_components: Dict[str, Any]):
    """Execute check dengan new progress tracker API"""
    try:
        _disable_operation_buttons(ui_components)
        
        # Show progress untuk check operation
        _show_progress_for_operation(ui_components, 'check_dataset')
        
        _log_to_ui(ui_components, "🔍 Memulai pengecekan dataset...", 'info')
        
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
        
        # Step 1: Find data location
        _update_progress_safe(ui_components, 'overall', 10, "🔍 Mencari lokasi data")
        
        data_location = get_best_data_location()
        _log_to_ui(ui_components, f"📁 Data location: {data_location}", 'info')
        
        # Step 2: Check raw dataset
        _update_progress_safe(ui_components, 'step', 30, "📊 Menganalisis raw dataset")
        
        raw_info = detect_split_structure(data_location)
        if raw_info['status'] == 'success':
            raw_images = raw_info.get('total_images', 0)
            raw_labels = raw_info.get('total_labels', 0)
            available_splits = raw_info.get('available_splits', [])
            
            _log_to_ui(ui_components, f"✅ Raw Dataset: {raw_images} gambar, {raw_labels} label", 'success')
            _log_to_ui(ui_components, f"📂 Available splits: {', '.join(available_splits) if available_splits else 'flat structure'}", 'info')
        else:
            _log_to_ui(ui_components, f"❌ Raw dataset tidak ditemukan: {raw_info.get('message', 'Unknown error')}", 'error')
        
        # Step 3: Check augmented dataset
        _update_progress_safe(ui_components, 'step', 60, "🔄 Mengecek augmented dataset")
        _check_augmented_dataset(ui_components, data_location)
        
        # Step 4: Check preprocessed dataset
        _update_progress_safe(ui_components, 'step', 80, "🔧 Mengecek preprocessed dataset")
        _check_preprocessed_dataset(ui_components, data_location)
        
        # Final summary
        ready_status = "✅ Siap" if raw_info['status'] == 'success' and raw_images > 0 else "❌ Tidak siap"
        _log_to_ui(ui_components, f"🎯 Status augmentasi: {ready_status}", 'success' if ready_status.startswith('✅') else 'warning')
        
        _complete_progress_safe(ui_components, "✅ Check dataset selesai")
        
    except Exception as e:
        error_msg = f"❌ Check error: {str(e)}"
        _log_to_ui(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)
    finally:
        _enable_operation_buttons(ui_components)

def execute_cleanup_with_progress(ui_components: Dict[str, Any]):
    """Execute cleanup dengan new progress tracker"""
    try:
        _disable_operation_buttons(ui_components)
        
        # Show progress untuk cleanup operation
        _show_progress_for_operation(ui_components, 'cleanup')
        
        _log_to_ui(ui_components, "🧹 Memulai cleanup dataset...", 'info')
        
        from smartcash.dataset.augmentor.service import create_service_from_ui
        
        service = create_service_from_ui(ui_components)
        
        # Progress updates dengan new API
        _update_progress_safe(ui_components, 'step', 30, "🔍 Mencari file augmented")
        
        result = service.cleanup_augmented_data(include_preprocessed=True)
        
        _update_progress_safe(ui_components, 'step', 80, "🧹 Menyelesaikan cleanup")
        
        if result.get('status') == 'success':
            total_deleted = result.get('total_deleted', 0)
            if total_deleted > 0:
                success_msg = f"✅ Cleanup berhasil: {total_deleted} file dihapus"
                _log_to_ui(ui_components, success_msg, 'success')
                _complete_progress_safe(ui_components, success_msg)
            else:
                info_msg = "💡 Cleanup selesai: tidak ada file untuk dihapus"
                _log_to_ui(ui_components, info_msg, 'info')
                _complete_progress_safe(ui_components, info_msg)
        elif result.get('status') == 'empty':
            info_msg = "💡 Cleanup selesai: tidak ada file untuk dihapus"
            _log_to_ui(ui_components, info_msg, 'info')
            _complete_progress_safe(ui_components, info_msg)
        else:
            warning_msg = f"⚠️ Cleanup warning: {result.get('message', 'Unknown issue')}"
            _log_to_ui(ui_components, warning_msg, 'warning')
            _complete_progress_safe(ui_components, "Cleanup completed with warnings")
        
    except Exception as e:
        error_msg = f"❌ Cleanup error: {str(e)}"
        _log_to_ui(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)
    finally:
        _enable_operation_buttons(ui_components)

def _create_service_safely(ui_components: Dict[str, Any]):
    """Create augmentation service dengan error handling"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        return create_service_from_ui(ui_components)
    except ImportError as e:
        _log_to_ui(ui_components, f"❌ Service import error: {str(e)}", 'error')
        return None
    except Exception as e:
        _log_to_ui(ui_components, f"❌ Service creation error: {str(e)}", 'error')
        return None

def _create_new_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk new progress tracker API"""
    import time
    last_update = {'time': 0, 'percentage': -1, 'step': ''}
    
    def callback(step: str, current: int, total: int, message: str):
        current_time = time.time()
        percentage = min(100, max(0, int((current / max(1, total)) * 100)))
        
        # Update lebih sering untuk step changes atau milestone
        should_update = (
            step != last_update['step'] or
            percentage in [0, 25, 50, 75, 100] or
            (current_time - last_update['time'] > 0.5)
        )
        
        if should_update:
            last_update.update({'time': current_time, 'percentage': percentage, 'step': step})
            
            # Log progress
            step_emoji = {'overall': '🎯', 'step': '📊', 'current': '⚡'}.get(step, '📈')
            progress_msg = f"{step_emoji} {step.title()}: {percentage}% - {message}"
            _log_to_ui(ui_components, progress_msg, 'info')
            
            # Update progress tracker dengan new API
            _update_progress_safe(ui_components, step, percentage, message)
    
    return callback

def _show_progress_for_operation(ui_components: Dict[str, Any], operation_name: str):
    """Show progress untuk operation tertentu dengan config dari defaults"""
    try:
        config = ui_components.get('config', {})
        progress_config = config.get('progress', {})
        operations_config = progress_config.get('operations', {})
        
        # Get operation config dengan fallback
        op_config = operations_config.get(operation_name, {
            'steps': ["prepare", "process", "complete"],
            'weights': {"prepare": 20, "process": 60, "complete": 20},
            'auto_hide': True
        })
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show(
                operation=operation_name.replace('_', ' ').title(),
                steps=op_config.get('steps'),
                step_weights=op_config.get('weights')
            )
        elif 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation_name)
            
    except Exception:
        # Fallback show
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show(operation_name.replace('_', ' ').title())

def _update_progress_safe(ui_components: Dict[str, Any], level: str, percentage: int, message: str):
    """Safe progress update dengan new API"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Try new API methods
            if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(percentage, message)
            elif level == 'step' and hasattr(progress_tracker, 'update_step'):
                progress_tracker.update_step(percentage, message)
            elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                progress_tracker.update_current(percentage, message)
            elif hasattr(progress_tracker, 'update'):
                # Fallback ke update generic
                progress_tracker.update(level, percentage, message)
    except Exception:
        pass

def _complete_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress complete dengan new API"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        elif 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
    except Exception:
        pass

def _error_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress error dengan new API"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        elif 'error_operation' in ui_components:
            ui_components['error_operation'](message)
    except Exception:
        pass

def _check_augmented_dataset(ui_components: Dict[str, Any], data_location: str):
    """Check augmented dataset dengan detail"""
    try:
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        
        aug_info = detect_split_structure(f"{data_location}/augmented")
        if aug_info['status'] == 'success' and aug_info.get('total_images', 0) > 0:
            aug_images = aug_info.get('total_images', 0)
            aug_splits = aug_info.get('available_splits', [])
            _log_to_ui(ui_components, f"✅ Augmented Dataset: {aug_images} file", 'success')
            if aug_splits:
                _log_to_ui(ui_components, f"📂 Augmented splits: {', '.join(aug_splits)}", 'info')
        else:
            _log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')
    except Exception:
        _log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file augmented", 'info')

def _check_preprocessed_dataset(ui_components: Dict[str, Any], data_location: str):
    """Check preprocessed dataset dengan detail"""
    try:
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        
        prep_info = detect_split_structure(f"{data_location}/preprocessed")
        if prep_info['status'] == 'success' and prep_info.get('total_images', 0) > 0:
            prep_images = prep_info.get('total_images', 0)
            prep_splits = prep_info.get('available_splits', [])
            _log_to_ui(ui_components, f"✅ Preprocessed Dataset: {prep_images} file", 'success')
            if prep_splits:
                _log_to_ui(ui_components, f"📂 Preprocessed splits: {', '.join(prep_splits)}", 'info')
        else:
            _log_to_ui(ui_components, "🔧 Preprocessed Dataset: Belum ada file preprocessed", 'info')
    except Exception:
        _log_to_ui(ui_components, "🔧 Preprocessed Dataset: Belum ada file preprocessed", 'info')

def _handle_service_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle service result dengan detailed feedback"""
    if result.get('status') == 'success':
        if operation == 'augmentation pipeline':
            total_generated = result.get('total_generated', 0)
            total_normalized = result.get('total_normalized', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"✅ Pipeline berhasil: Generated {total_generated}, Normalized {total_normalized}, Time {processing_time:.1f}s"
            
            # Log additional details
            if 'aug_result' in result:
                aug_result = result['aug_result']
                _log_to_ui(ui_components, f"📈 Augmentation success rate: {aug_result.get('success_rate', 0):.1f}%", 'info')
            
            if 'norm_result' in result:
                norm_result = result['norm_result']
                target_dir = norm_result.get('target_dir', 'data/preprocessed')
                _log_to_ui(ui_components, f"📁 Preprocessed files saved to: {target_dir}", 'info')
        else:
            success_msg = f"✅ {operation.title()} berhasil"
        
        _log_to_ui(ui_components, success_msg, 'success')
        _complete_progress_safe(ui_components, success_msg)
        
    else:
        error_msg = f"❌ {operation.title()} gagal: {result.get('message', 'Unknown error')}"
        _log_to_ui(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)

def _disable_operation_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons dengan visual feedback"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button']
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            button.disabled = True
            # Update description dengan processing indicator
            if hasattr(button, 'description'):
                original_desc = getattr(button, '_original_description', button.description)
                if not hasattr(button, '_original_description'):
                    button._original_description = button.description
                button.description = f"⏳ {original_desc}"

def _enable_operation_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons dengan restore description"""
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

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log ke UI dengan fallback chain"""
    try:
        # Priority 1: UI Logger
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Priority 2: Log widget
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            from IPython.display import display, HTML
            color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = color_map.get(level, '#007bff')
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
            
            with widget:
                display(HTML(html))
            return
            
    except Exception:
        pass
    
    # Fallback
    print(message)