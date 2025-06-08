"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Enhanced handlers dengan backend integration dan button manager
"""

from typing import Dict, Any

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan backend integration dan enhanced progress tracking"""
    
    # Setup config handler
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup backend integrator
    _setup_backend_integration(ui_components)
    
    # Setup enhanced operation handlers dengan button manager
    _setup_enhanced_operation_handlers_with_backend(ui_components)
    _setup_config_handlers_with_validation(ui_components)
    
    return ui_components

def _setup_backend_integration(ui_components: Dict[str, Any]):
    """Setup backend service integration"""
    try:
        from smartcash.ui.dataset.augmentation.utils.backend_communicator import create_service_integrator
        from smartcash.ui.dataset.augmentation.utils.button_manager import OperationManager
        
        # Create service integrator
        service_integrator = create_service_integrator(ui_components)
        ui_components['service_integrator'] = service_integrator
        ui_components['backend_communicator'] = service_integrator.get_communicator()
        
        # Create operation manager untuk button coordination
        operation_manager = OperationManager(ui_components)
        ui_components['operation_manager'] = operation_manager
        
    except ImportError:
        _log_ui(ui_components, "⚠️ Backend integration tidak tersedia - menggunakan fallback", 'warning')

def _setup_enhanced_operation_handlers_with_backend(ui_components: Dict[str, Any]):
    """Setup operation handlers dengan backend service integration"""
    
    def augment_handler(button):
        """Enhanced augmentation handler dengan backend integration"""
        _clear_outputs(ui_components)
        
        # Validate form inputs terlebih dahulu
        validation_result = _validate_inputs(ui_components)
        if not validation_result['valid']:
            _show_validation_errors(ui_components, validation_result)
            return
        
        # Execute dengan operation manager
        operation_manager = ui_components.get('operation_manager')
        if operation_manager:
            operation_manager.execute_operation(
                "Augmentation Pipeline", 
                "augment_button", 
                _execute_augmentation_with_backend,
                ui_components
            )
        else:
            # Fallback tanpa operation manager
            _execute_augmentation_with_backend(ui_components)
    
    def enhanced_check_handler(button):
        """Enhanced check handler dengan dataset comparison"""
        _clear_outputs(ui_components)
        
        operation_manager = ui_components.get('operation_manager')
        if operation_manager:
            operation_manager.execute_operation(
                "Dataset Check", 
                "check_button", 
                _execute_enhanced_check_with_backend,
                ui_components
            )
        else:
            _execute_enhanced_check_with_backend(ui_components)
    
    def cleanup_handler(button):
        """Cleanup handler dengan confirmation dialog"""
        _clear_outputs(ui_components)
        
        def confirm_cleanup(confirm_button):
            operation_manager = ui_components.get('operation_manager')
            if operation_manager:
                operation_manager.execute_operation(
                    "Cleanup Dataset", 
                    "cleanup_button", 
                    _execute_cleanup_with_backend,
                    ui_components
                )
            else:
                _execute_cleanup_with_backend(ui_components)
        
        def cancel_cleanup(cancel_button):
            _log_ui(ui_components, "❌ Cleanup dibatalkan", 'info')
        
        # Show confirmation dengan dialog utils
        try:
            from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_cleanup_confirmation
            show_cleanup_confirmation(ui_components, confirm_cleanup, cancel_cleanup)
        except ImportError:
            # Direct execute jika dialog tidak tersedia
            confirm_cleanup(None)
    
    # Bind handlers
    handlers = {
        'augment_button': augment_handler,
        'check_button': enhanced_check_handler, 
        'cleanup_button': cleanup_handler
    }
    
    for button_name, handler in handlers.items():
        button = ui_components.get(button_name)
        if button and hasattr(button, 'on_click'):
            button.on_click(handler)

def _execute_augmentation_with_backend(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Execute augmentation menggunakan backend service"""
    service_integrator = ui_components.get('service_integrator')
    if not service_integrator:
        return _execute_augmentation_fallback(ui_components)
    
    try:
        # Create service
        service = service_integrator.create_augmentation_service()
        if not service:
            return {'status': 'error', 'message': 'Gagal membuat augmentation service'}
        
        # Get target split
        target_split = _get_target_split_safe(ui_components)
        
        # Execute dengan progress tracking
        result = service_integrator.execute_with_progress(
            "Augmentation Pipeline",
            service.run_full_augmentation_pipeline,
            target_split=target_split
        )
        
        return result or {'status': 'error', 'message': 'Service returned no result'}
        
    except Exception as e:
        return {'status': 'error', 'message': f'Backend service error: {str(e)}'}

def _execute_enhanced_check_with_backend(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Execute enhanced check menggunakan backend service"""
    service_integrator = ui_components.get('service_integrator')
    if not service_integrator:
        return _execute_check_fallback(ui_components)
    
    try:
        service = service_integrator.create_augmentation_service()
        if not service:
            return {'status': 'error', 'message': 'Gagal membuat service untuk check'}
        
        target_split = _get_target_split_safe(ui_components)
        
        result = service_integrator.execute_with_progress(
            "Dataset Check",
            service.check_dataset_readiness,
            target_split
        )
        
        return result or {'status': 'error', 'message': 'Check service returned no result'}
        
    except Exception as e:
        return {'status': 'error', 'message': f'Check service error: {str(e)}'}

def _execute_cleanup_with_backend(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Execute cleanup menggunakan backend service"""
    service_integrator = ui_components.get('service_integrator')
    if not service_integrator:
        return _execute_cleanup_fallback(ui_components)
    
    try:
        service = service_integrator.create_augmentation_service()
        if not service:
            return {'status': 'error', 'message': 'Gagal membuat service untuk cleanup'}
        
        result = service_integrator.execute_with_progress(
            "Cleanup Dataset",
            service.cleanup_augmented_data,
            include_preprocessed=True
        )
        
        return result or {'status': 'error', 'message': 'Cleanup service returned no result'}
        
    except Exception as e:
        return {'status': 'error', 'message': f'Cleanup service error: {str(e)}'}

def _setup_config_handlers_with_validation(ui_components: Dict[str, Any]):
    """Setup config handlers dengan enhanced validation"""
    
    def save_config(button=None):
        _clear_outputs(ui_components)
        
        # Validate inputs terlebih dahulu
        validation_result = _validate_inputs(ui_components)
        if not validation_result['valid']:
            _show_validation_errors(ui_components, validation_result)
            return
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                
                success = config_handler.save_config(ui_components)
                # Config handler sudah log ke UI
                
            else:
                _log_ui(ui_components, "❌ Config handler tidak tersedia", 'error')
        except Exception as e:
            _log_ui(ui_components, f"❌ Save error: {str(e)}", 'error')
    
    def reset_config(button=None):
        _clear_outputs(ui_components)
        
        def confirm_reset(confirm_button):
            try:
                config_handler = ui_components.get('config_handler')
                if config_handler:
                    if hasattr(config_handler, 'set_ui_components'):
                        config_handler.set_ui_components(ui_components)
                    
                    success = config_handler.reset_config(ui_components)
                    # Config handler sudah log ke UI
                    
                else:
                    _log_ui(ui_components, "❌ Config handler tidak tersedia", 'error')
            except Exception as e:
                _log_ui(ui_components, f"❌ Reset error: {str(e)}", 'error')
        
        def cancel_reset(cancel_button):
            _log_ui(ui_components, "❌ Reset dibatalkan", 'info')
        
        # Show reset confirmation
        try:
            from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_reset_confirmation
            show_reset_confirmation(ui_components, confirm_reset, cancel_reset)
        except ImportError:
            # Direct execute jika dialog tidak tersedia
            confirm_reset(None)
    
    # Bind config handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    
    if save_button and hasattr(save_button, 'on_click'):
        save_button.on_click(save_config)
    if reset_button and hasattr(reset_button, 'on_click'):
        reset_button.on_click(reset_config)

# Validation dan utility functions
def _validate_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate form inputs dengan detailed feedback"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs
        return validate_form_inputs(ui_components)
    except ImportError:
        return {'valid': True, 'errors': [], 'warnings': []}

def _show_validation_errors(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Show validation errors menggunakan dialog utils"""
    try:
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_validation_errors_dialog
        show_validation_errors_dialog(ui_components, validation_result)
    except ImportError:
        # Fallback ke UI log
        for error in validation_result.get('errors', []):
            _log_ui(ui_components, error, 'error')
        for warning in validation_result.get('warnings', []):
            _log_ui(ui_components, warning, 'warning')

def _get_target_split_safe(ui_components: Dict[str, Any]) -> str:
    """Get target split dengan safe default"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import get_widget_value_safe
        return get_widget_value_safe(ui_components, 'target_split', 'train')
    except ImportError:
        target_split_widget = ui_components.get('target_split')
        if target_split_widget and hasattr(target_split_widget, 'value'):
            return getattr(target_split_widget, 'value', 'train')
        return 'train'

# Fallback implementations tanpa backend service
def _execute_augmentation_fallback(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback augmentation tanpa backend service"""
    _log_ui(ui_components, "⚠️ Backend service tidak tersedia - menggunakan fallback", 'warning')
    return {'status': 'error', 'message': 'Backend service tidak tersedia'}

def _execute_check_fallback(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback check tanpa backend service"""
    _log_ui(ui_components, "⚠️ Check service tidak tersedia", 'warning')
    return {'status': 'error', 'message': 'Check service tidak tersedia'}

def _execute_cleanup_fallback(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback cleanup tanpa backend service"""
    _log_ui(ui_components, "⚠️ Cleanup service tidak tersedia", 'warning')
    return {'status': 'error', 'message': 'Cleanup service tidak tersedia'}

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear outputs dan reset progress"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
        clear_ui_outputs(ui_components)
    except ImportError:
        # Fallback manual clear
        output_keys = ['log_output', 'status', 'confirmation_area']
        for key in output_keys:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'clear_output'):
                try:
                    widget.clear_output(wait=True)
                except Exception:
                    pass
    
    # Reset progress tracker
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    except Exception:
        pass

def _log_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log ke UI dengan fallback chain"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, message, level)
    except ImportError:
        # Fallback manual logging
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
        
        # Final fallback
        emoji_map = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        print(f"{emoji_map.get(level, 'ℹ️')} {message}")