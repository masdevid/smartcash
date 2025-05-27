"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed main handler dengan proper button registration tanpa UI clearing
"""

from typing import Dict, Any

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler augmentation tanpa UI clearing"""
    from .button_handler import create_button_handler
    from .operation_handlers import create_augmentation_handler
    
    button_handler = create_button_handler(ui_components)
    operation_handler = create_augmentation_handler(ui_components)
    
    button_handler.execute_operation('augmentation', operation_handler.execute)

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler check dataset"""
    from .button_handler import create_button_handler
    from .operation_handlers import create_check_handler
    
    button_handler = create_button_handler(ui_components)
    operation_handler = create_check_handler(ui_components)
    
    button_handler.execute_operation('check', operation_handler.execute)

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler cleanup dengan confirmation"""
    from .confirmation_handler import create_confirmation_handler
    
    confirmation_handler = create_confirmation_handler(ui_components)
    
    def confirm_cleanup(confirm_button):
        from .button_handler import create_button_handler
        from .operation_handlers import create_cleanup_handler
        
        button_handler = create_button_handler(ui_components)
        operation_handler = create_cleanup_handler(ui_components)
        button_handler.execute_operation('cleanup', operation_handler.execute)
    
    confirmation_handler.show_cleanup_confirmation(confirm_cleanup)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler save config tanpa UI clearing"""
    from .button_handler import create_button_handler
    from .operation_handlers import create_config_handler_ops
    
    button_handler = create_button_handler(ui_components)
    config_handler = create_config_handler_ops(ui_components)
    
    button_handler.execute_config('save', config_handler.save_config)

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler reset config tanpa UI clearing"""
    from .button_handler import create_button_handler
    from .operation_handlers import create_config_handler_ops
    
    button_handler = create_button_handler(ui_components)
    config_handler = create_config_handler_ops(ui_components)
    
    button_handler.execute_config('reset', config_handler.reset_config)

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers dengan proper button registration"""
    try:
        from .button_handler import create_button_handler
        
        # Create button handler
        button_handler = create_button_handler(ui_components)
        ui_components['button_state_manager'] = button_handler.manager
        
        # Handler mapping
        button_handlers = {
            'augment_button': handle_augmentation_button_click,
            'check_button': handle_check_dataset_button_click,
            'cleanup_button': handle_cleanup_button_click,
            'save_button': handle_save_config_button_click,
            'reset_button': handle_reset_config_button_click
        }
        
        # Register handlers dengan safe approach
        registered_count = 0
        for button_key, handler_func in button_handlers.items():
            try:
                button = ui_components.get(button_key)
                if button and hasattr(button, 'on_click'):
                    # Clear existing handlers untuk prevent double registration
                    if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
                        button._click_handlers.callbacks.clear()
                    
                    # Register new handler dengan proper closure
                    def create_handler(handler, ui_comps):
                        return lambda b: handler(ui_comps, b)
                    
                    button.on_click(create_handler(handler_func, ui_components))
                    registered_count += 1
            except Exception as e:
                if 'logger' in ui_components:
                    ui_components['logger'].debug(f"⚠️ Failed to register {button_key}: {str(e)}")
                continue
        
        # Update registration info
        ui_components['registered_handlers'] = {
            'total': registered_count,
            'handlers': list(button_handlers.keys()),
            'status': 'success' if registered_count > 0 else 'partial',
            'button_state_manager_active': True
        }
        
        # Log success
        if 'logger' in ui_components:
            ui_components['logger'].info(f"✅ {registered_count} handlers registered successfully")
        
        return ui_components
        
    except Exception as e:
        # Fallback registration info
        ui_components['registered_handlers'] = {
            'total': 0,
            'handlers': [],
            'status': 'error',
            'error': str(e),
            'button_state_manager_active': False
        }
        
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Handler registration error: {str(e)}")
        
        return ui_components