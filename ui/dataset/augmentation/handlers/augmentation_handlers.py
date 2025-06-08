"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Fixed handlers dengan proper service integration dan error handling
"""

from typing import Dict, Any

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan service integration yang benar"""
    
    # CRITICAL: Setup config handler dengan UI logger
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup operation handlers dengan safe execution
    _setup_operation_handlers_safe(ui_components)
    
    # Setup config handlers
    _setup_config_handlers_safe(ui_components)
    
    return ui_components

def _setup_operation_handlers_safe(ui_components: Dict[str, Any]):
    """Setup operation handlers dengan safe execution dan proper imports"""
    
    def augment_handler(button):
        """Augmentation handler dengan service integration"""
        _clear_outputs_safe(ui_components)
        _log_to_ui_safe(ui_components, "🚀 Memulai pipeline augmentasi...", 'info')
        
        try:
            # Import service dengan error handling
            service = _create_service_safe(ui_components)
            if not service:
                _log_to_ui_safe(ui_components, "❌ Gagal membuat augmentation service", 'error')
                return
            
            # Execute dengan progress callback
            target_split = _get_target_split_safe(ui_components)
            progress_callback = _create_progress_callback_safe(ui_components)
            
            result = service.run_full_augmentation_pipeline(
                target_split=target_split,
                progress_callback=progress_callback
            )
            
            _handle_service_result(ui_components, result, 'augmentation')
            
        except Exception as e:
            _log_to_ui_safe(ui_components, f"❌ Pipeline error: {str(e)}", 'error')
    
    def check_handler(button):
        """Check dataset handler dengan dataset detector"""
        _clear_outputs_safe(ui_components)
        _log_to_ui_safe(ui_components, "🔍 Memulai pengecekan dataset...", 'info')
        
        try:
            from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
            from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
            
            data_location = get_best_data_location()
            _log_to_ui_safe(ui_components, f"📁 Data location: {data_location}", 'info')
            
            # Check raw dataset
            raw_info = detect_split_structure(data_location)
            if raw_info['status'] == 'success':
                raw_images = raw_info.get('total_images', 0)
                raw_labels = raw_info.get('total_labels', 0)
                available_splits = raw_info.get('available_splits', [])
                
                _log_to_ui_safe(ui_components, f"✅ Raw Dataset: {raw_images} gambar, {raw_labels} label", 'success')
                if available_splits:
                    _log_to_ui_safe(ui_components, f"📂 Available splits: {', '.join(available_splits)}", 'info')
                
                ready_status = "✅ Siap untuk augmentasi" if raw_images > 0 else "❌ Dataset kosong"
                _log_to_ui_safe(ui_components, ready_status, 'success' if raw_images > 0 else 'warning')
            else:
                _log_to_ui_safe(ui_components, f"❌ Dataset tidak ditemukan: {raw_info.get('message', 'Unknown error')}", 'error')
            
        except Exception as e:
            _log_to_ui_safe(ui_components, f"❌ Check error: {str(e)}", 'error')
    
    def cleanup_handler(button):
        """Cleanup handler dengan confirmation dialog"""
        _clear_outputs_safe(ui_components)
        
        def confirm_cleanup(confirm_button):
            _log_to_ui_safe(ui_components, "🧹 Memulai cleanup dataset...", 'info')
            
            try:
                service = _create_service_safe(ui_components)
                if service:
                    result = service.cleanup_augmented_data(include_preprocessed=True)
                    
                    if result.get('status') == 'success':
                        total_deleted = result.get('total_deleted', 0)
                        message = f"✅ Cleanup berhasil: {total_deleted} file dihapus" if total_deleted > 0 else "💡 Tidak ada file untuk dihapus"
                        _log_to_ui_safe(ui_components, message, 'success' if total_deleted > 0 else 'info')
                    else:
                        _log_to_ui_safe(ui_components, f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}", 'error')
                else:
                    _log_to_ui_safe(ui_components, "❌ Service tidak tersedia untuk cleanup", 'error')
                    
            except Exception as e:
                _log_to_ui_safe(ui_components, f"❌ Cleanup error: {str(e)}", 'error')
        
        # Show confirmation atau direct execute
        try:
            from smartcash.ui.components.dialogs import show_destructive_confirmation
            show_destructive_confirmation(
                title="Konfirmasi Cleanup",
                message="Hapus semua file augmented?\n\n⚠️ Tindakan tidak dapat dibatalkan!",
                item_name="file augmented",
                on_confirm=confirm_cleanup,
                on_cancel=lambda b: _log_to_ui_safe(ui_components, "❌ Cleanup dibatalkan", 'info')
            )
        except ImportError:
            # Direct execute jika dialog tidak tersedia
            confirm_cleanup(None)
    
    # Bind handlers dengan safe access
    button_handlers = {
        'augment_button': augment_handler,
        'check_button': check_handler,
        'cleanup_button': cleanup_handler
    }
    
    for button_name, handler in button_handlers.items():
        button = ui_components.get(button_name)
        if button and hasattr(button, 'on_click'):
            button.on_click(handler)

def _setup_config_handlers_safe(ui_components: Dict[str, Any]):
    """Setup config handlers dengan safe execution"""
    
    def save_config(button=None):
        _clear_outputs_safe(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.save_config(ui_components)
            else:
                _log_to_ui_safe(ui_components, "❌ Config handler tidak tersedia", 'error')
        except Exception as e:
            _log_to_ui_safe(ui_components, f"❌ Error save: {str(e)}", 'error')
    
    def reset_config(button=None):
        _clear_outputs_safe(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.reset_config(ui_components)
            else:
                _log_to_ui_safe(ui_components, "❌ Config handler tidak tersedia", 'error')
        except Exception as e:
            _log_to_ui_safe(ui_components, f"❌ Error reset: {str(e)}", 'error')
    
    # Bind config handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    
    if save_button and hasattr(save_button, 'on_click'):
        save_button.on_click(save_config)
    if reset_button and hasattr(reset_button, 'on_click'):
        reset_button.on_click(reset_config)

def _create_service_safe(ui_components: Dict[str, Any]):
    """Create augmentation service dengan safe import"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        return create_service_from_ui(ui_components)
    except ImportError as e:
        _log_to_ui_safe(ui_components, f"❌ Service import error: {str(e)}", 'error')
        return None
    except Exception as e:
        _log_to_ui_safe(ui_components, f"❌ Service creation error: {str(e)}", 'error')
        return None

def _create_progress_callback_safe(ui_components: Dict[str, Any]):
    """Create progress callback dengan safe execution"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            percentage = min(100, max(0, int((current / max(1, total)) * 100)))
            step_emoji = {'overall': '🎯', 'step': '📊', 'current': '⚡'}.get(step, '📈')
            progress_msg = f"{step_emoji} {step.title()}: {percentage}% - {message}"
            _log_to_ui_safe(ui_components, progress_msg, 'info')
        except Exception:
            pass
    
    return progress_callback

def _handle_service_result(ui_components: Dict[str, Any], result: Dict[str, Any], operation: str):
    """Handle service result dengan detailed feedback"""
    if result.get('status') == 'success':
        if operation == 'augmentation':
            total_generated = result.get('total_generated', 0)
            total_normalized = result.get('total_normalized', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"✅ Pipeline berhasil: Generated {total_generated}, Normalized {total_normalized}, Time {processing_time:.1f}s"
        else:
            success_msg = f"✅ {operation.title()} berhasil"
        
        _log_to_ui_safe(ui_components, success_msg, 'success')
    else:
        error_msg = f"❌ {operation.title()} gagal: {result.get('message', 'Unknown error')}"
        _log_to_ui_safe(ui_components, error_msg, 'error')

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'

def _clear_outputs_safe(ui_components: Dict[str, Any]):
    """Clear outputs dengan safe execution"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
        
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output(wait=True)
    except Exception:
        pass

def _log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log ke UI dengan safe fallback chain"""
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
    
    # Fallback print
    print(message)