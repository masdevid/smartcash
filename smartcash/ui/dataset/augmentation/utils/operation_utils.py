"""
File: smartcash/ui/dataset/augmentation/utils/operation_utils.py
Deskripsi: Enhanced operation utilities dengan granular progress dan dataset comparison
"""

from typing import Dict, Any

def execute_augmentation(ui_components: Dict[str, Any]):
    """Execute augmentation dengan granular progress tracking"""
    try:
        _disable_operation_buttons(ui_components)
        _show_progress_for_operation(ui_components, 'augmentation')
        _log_to_ui(ui_components, "🚀 Memulai pipeline augmentasi...", 'info')
        
        service = _create_service_safely(ui_components)
        if not service:
            _log_to_ui(ui_components, "❌ Gagal membuat augmentation service", 'error')
            return
        
        target_split = _get_target_split_safe(ui_components)
        _log_to_ui(ui_components, f"📂 Target split: {target_split}", 'info')
        
        result = service.run_full_augmentation_pipeline(
            target_split=target_split,
            progress_callback=_create_progress_callback(ui_components)
        )
        
        _handle_service_result(ui_components, result, 'augmentation pipeline')
        
    except Exception as e:
        error_msg = f"❌ Pipeline error: {str(e)}"
        _log_to_ui(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)
    finally:
        _enable_operation_buttons(ui_components)

def execute_enhanced_check(ui_components: Dict[str, Any]):
    """🆕 Execute enhanced check dengan raw vs preprocessed comparison"""
    try:
        _disable_operation_buttons(ui_components)
        _show_progress_for_operation(ui_components, 'check_dataset')
        _log_to_ui(ui_components, "🔍 Memulai pengecekan dataset comprehensive...", 'info')
        
        service = _create_service_safely(ui_components)
        if not service:
            _log_to_ui(ui_components, "❌ Gagal membuat service untuk check", 'error')
            return
        
        target_split = _get_target_split_safe(ui_components)
        
        # Execute enhanced dataset check
        result = service.check_dataset_readiness(target_split)
        
        # Handle enhanced results
        if result.get('status') == 'success':
            _log_enhanced_check_results(ui_components, result)
            
            if result.get('ready_for_augmentation'):
                success_msg = "✅ Dataset siap untuk augmentasi"
                _log_to_ui(ui_components, success_msg, 'success')
                _complete_progress_safe(ui_components, success_msg)
            else:
                warning_msg = "⚠️ Dataset perlu preprocessing sebelum augmentasi"
                _log_to_ui(ui_components, warning_msg, 'warning')
                _complete_progress_safe(ui_components, warning_msg)
        else:
            error_msg = f"❌ Check gagal: {result.get('message', 'Unknown error')}"
            _log_to_ui(ui_components, error_msg, 'error')
            _error_progress_safe(ui_components, error_msg)
        
    except Exception as e:
        error_msg = f"❌ Check error: {str(e)}"
        _log_to_ui(ui_components, error_msg, 'error')
        _error_progress_safe(ui_components, error_msg)
    finally:
        _enable_operation_buttons(ui_components)

def execute_cleanup_with_progress(ui_components: Dict[str, Any]):
    """Execute cleanup dengan granular progress tracking"""
    try:
        _disable_operation_buttons(ui_components)
        _show_progress_for_operation(ui_components, 'cleanup')
        _log_to_ui(ui_components, "🧹 Memulai cleanup dataset...", 'info')
        
        service = _create_service_safely(ui_components)
        if not service:
            _log_to_ui(ui_components, "❌ Gagal membuat service untuk cleanup", 'error')
            return
        
        result = service.cleanup_augmented_data(include_preprocessed=True)
        
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

def _log_enhanced_check_results(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """🆕 Log enhanced check results dengan detail comparison"""
    data_location = result.get('data_location', 'Unknown')
    comparison = result.get('comparison', {})
    
    _log_to_ui(ui_components, f"📁 Data location: {data_location}", 'info')
    
    # Raw dataset info
    raw_info = result.get('raw_dataset', {})
    if raw_info.get('status') == 'success':
        total_raw = raw_info.get('total_images', 0)
        splits = raw_info.get('available_splits', [])
        _log_to_ui(ui_components, f"✅ Raw Dataset: {total_raw} gambar di {len(splits)} splits", 'success')
        if splits:
            _log_to_ui(ui_components, f"📂 Available splits: {', '.join(splits)}", 'info')
    else:
        _log_to_ui(ui_components, f"❌ Raw dataset: {raw_info.get('message', 'Not found')}", 'error')
    
    # Preprocessed comparison
    if comparison.get('preprocessed_exists'):
        split_comparison = comparison.get('split_comparison', {})
        total_prep = sum(details.get('preprocessed_images', 0) for details in split_comparison.values())
        _log_to_ui(ui_components, f"✅ Preprocessed Dataset: {total_prep} file", 'success')
        
        # Per-split details
        for split, details in split_comparison.items():
            raw_count = details.get('raw_images', 0)
            prep_count = details.get('preprocessed_images', 0)
            if details.get('needs_preprocessing'):
                _log_to_ui(ui_components, f"🔄 {split}: {raw_count} raw → perlu preprocessing", 'warning')
            elif prep_count > 0:
                _log_to_ui(ui_components, f"✅ {split}: {prep_count} preprocessed siap", 'success')
    else:
        _log_to_ui(ui_components, "🔄 Preprocessed Dataset: Belum tersedia", 'warning')
    
    # Augmented dataset info
    aug_info = result.get('augmented_dataset', {})
    if aug_info.get('status') == 'success' and aug_info.get('total_images', 0) > 0:
        aug_images = aug_info.get('total_images', 0)
        _log_to_ui(ui_components, f"✅ Augmented Dataset: {aug_images} file", 'success')
    else:
        _log_to_ui(ui_components, "🔄 Augmented Dataset: Belum ada file", 'info')
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    for rec in recommendations[:3]:  # Show top 3 recommendations
        _log_to_ui(ui_components, f"💡 {rec}", 'info')

# Helper functions
def _create_service_safely(ui_components: Dict[str, Any]):
    """Create augmentation service dengan error handling"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        return create_service_from_ui(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"❌ Service creation error: {str(e)}", 'error')
        return None

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback menggunakan existing progress_utils"""
    from smartcash.ui.dataset.augmentation.utils.progress_utils import create_progress_manager
    progress_manager = create_progress_manager(ui_components)
    return progress_manager.create_progress_callback()

def _show_progress_for_operation(ui_components: Dict[str, Any], operation_name: str):
    """Show progress menggunakan existing progress_utils"""
    from smartcash.ui.dataset.augmentation.utils.progress_utils import create_progress_manager
    progress_manager = create_progress_manager(ui_components)
    progress_manager.show_for_operation(operation_name)

def _update_progress_safe(ui_components: Dict[str, Any], level: str, percentage: int, message: str):
    """Update progress menggunakan existing progress_utils"""
    from smartcash.ui.dataset.augmentation.utils.progress_utils import create_progress_manager
    progress_manager = create_progress_manager(ui_components)
    progress_manager.update_tracker(level, percentage, message)

def _complete_progress_safe(ui_components: Dict[str, Any], message: str):
    """Complete progress menggunakan existing progress_utils"""
    from smartcash.ui.dataset.augmentation.utils.progress_utils import create_progress_manager
    progress_manager = create_progress_manager(ui_components)
    progress_manager.complete_operation(message)

def _error_progress_safe(ui_components: Dict[str, Any], message: str):
    """Error progress menggunakan existing progress_utils"""
    from smartcash.ui.dataset.augmentation.utils.progress_utils import create_progress_manager
    progress_manager = create_progress_manager(ui_components)
    progress_manager.error_operation(message)

def _handle_service_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle service result dengan detailed feedback"""
    if result.get('status') == 'success':
        if operation == 'augmentation pipeline':
            total_generated = result.get('total_generated', 0)
            total_normalized = result.get('total_normalized', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"✅ Pipeline berhasil: Generated {total_generated}, Normalized {total_normalized}, Time {processing_time:.1f}s"
            
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
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
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
    
    print(message)