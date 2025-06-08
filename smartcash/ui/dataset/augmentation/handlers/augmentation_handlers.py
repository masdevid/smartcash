"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Updated handlers dengan unified progress tracking integration
"""

from typing import Dict, Any

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan unified progress tracking"""
    
    # Setup config handler
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup unified progress manager
    progress_manager = _create_progress_manager(ui_components)
    ui_components['progress_manager'] = progress_manager
    
    # Setup handlers
    _setup_operation_handlers(ui_components, progress_manager)
    _setup_config_handlers(ui_components)
    
    return ui_components

def _create_progress_manager(ui_components: Dict[str, Any]):
    """Create unified progress manager"""
    try:
        from smartcash.ui.dataset.augmentation.utils.progress_utils import create_unified_progress_manager
        return create_unified_progress_manager(ui_components)
    except ImportError:
        return None

def _setup_operation_handlers(ui_components: Dict[str, Any], progress_manager):
    """Setup operation handlers dengan unified progress"""
    
    def augment_handler(button):
        """Augmentation handler dengan unified progress tracking"""
        _clear_outputs(ui_components)
        
        try:
            if progress_manager:
                progress_manager.start_operation("Dataset Augmentation")
            
            # Create service dengan UI integration
            from smartcash.dataset.augmentor.service import create_service_from_ui
            service = create_service_from_ui(ui_components)
            
            # Execute pipeline dengan progress callback
            target_split = _get_target_split(ui_components)
            progress_callback = progress_manager.create_progress_callback() if progress_manager else None
            
            result = service.run_full_augmentation_pipeline(target_split, progress_callback)
            
            # Progress manager handles completion via service
            
        except Exception as e:
            if progress_manager:
                progress_manager.error_operation(f"Pipeline error: {str(e)}")
            else:
                _log_ui(ui_components, f"❌ Pipeline error: {str(e)}", 'error')
    
    def check_handler(button):
        """Check dataset handler"""
        _clear_outputs(ui_components)
        
        try:
            if progress_manager:
                progress_manager.start_operation("Dataset Check")
                progress_manager._update_progress_with_throttling('overall', 10, 100, "Mencari lokasi data")
            
            from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
            from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
            
            data_location = get_best_data_location()
            
            if progress_manager:
                progress_manager._update_progress_with_throttling('overall', 50, 100, "Menganalisis dataset")
            
            raw_info = detect_split_structure(data_location)
            
            if raw_info['status'] == 'success':
                raw_images = raw_info.get('total_images', 0)
                splits = raw_info.get('available_splits', [])
                
                if progress_manager:
                    progress_manager.complete_operation(
                        f"Dataset siap: {raw_images} gambar di {len(splits)} split"
                    )
                else:
                    _log_ui(ui_components, f"✅ Dataset: {raw_images} gambar", 'success')
            else:
                error_msg = f"Dataset tidak ditemukan: {raw_info.get('message')}"
                if progress_manager:
                    progress_manager.error_operation(error_msg)
                else:
                    _log_ui(ui_components, f"❌ {error_msg}", 'error')
            
        except Exception as e:
            if progress_manager:
                progress_manager.error_operation(f"Check error: {str(e)}")
            else:
                _log_ui(ui_components, f"❌ Check error: {str(e)}", 'error')
    
    def cleanup_handler(button):
        """Cleanup handler dengan confirmation"""
        _clear_outputs(ui_components)
        
        def confirm_cleanup(confirm_button):
            try:
                if progress_manager:
                    progress_manager.start_operation("Cleanup Dataset")
                
                from smartcash.dataset.augmentor.service import create_service_from_ui
                service = create_service_from_ui(ui_components)
                result = service.cleanup_augmented_data(include_preprocessed=True)
                
                # Progress handled by service
                
            except Exception as e:
                if progress_manager:
                    progress_manager.error_operation(f"Cleanup error: {str(e)}")
                else:
                    _log_ui(ui_components, f"❌ Cleanup error: {str(e)}", 'error')
        
        # Show confirmation
        try:
            from smartcash.ui.components.dialogs import show_destructive_confirmation
            show_destructive_confirmation(
                "Konfirmasi Cleanup",
                "Hapus semua file augmented?\n\n⚠️ Tidak dapat dibatalkan!",
                "file augmented",
                confirm_cleanup,
                lambda b: _log_ui(ui_components, "❌ Cleanup dibatalkan", 'info')
            )
        except ImportError:
            confirm_cleanup(None)
    
    # Bind handlers
    handlers = {
        'augment_button': augment_handler,
        'check_button': check_handler, 
        'cleanup_button': cleanup_handler
    }
    
    for button_name, handler in handlers.items():
        button = ui_components.get(button_name)
        if button and hasattr(button, 'on_click'):
            button.on_click(handler)

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup config handlers"""
    
    def save_config(button=None):
        _clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.save_config(ui_components)
            else:
                _log_ui(ui_components, "❌ Config handler tidak tersedia", 'error')
        except Exception as e:
            _log_ui(ui_components, f"❌ Save error: {str(e)}", 'error')
    
    def reset_config(button=None):
        _clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.reset_config(ui_components)
            else:
                _log_ui(ui_components, "❌ Config handler tidak tersedia", 'error')
        except Exception as e:
            _log_ui(ui_components, f"❌ Reset error: {str(e)}", 'error')
    
    # Bind config handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    
    if save_button and hasattr(save_button, 'on_click'):
        save_button.on_click(save_config)
    if reset_button and hasattr(reset_button, 'on_click'):
        reset_button.on_click(reset_config)

def _get_target_split(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        return getattr(target_split_widget, 'value', 'train')
    return 'train'

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear outputs dan reset progress"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
        
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output(wait=True)
    except Exception:
        pass

def _log_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log ke UI dengan fallback"""
    try:
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            from IPython.display import display, HTML
            colors = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = colors.get(level, '#007bff')
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
            
            with widget:
                display(HTML(html))
            return
    except Exception:
        pass
    
    print(message)