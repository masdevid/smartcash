"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Handlers dengan backend integration dan dialog confirmation
"""

from typing import Dict, Any

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan backend integration dan dialog system"""
    
    # Setup config handler dengan UI integration
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup operation handlers
    _setup_operation_handlers(ui_components, config)
    _setup_config_handlers(ui_components, config)
    
    return ui_components

def _setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup operation handlers dengan backend service integration"""
    
    def execute_augmentation_pipeline(button=None):
        """Execute augmentation dengan progress tracking dan validation"""
        from smartcash.ui.dataset.augmentation.utils.operation_handlers import handle_augmentation_execution
        handle_augmentation_execution(ui_components)
    
    def execute_dataset_check(button=None):
        """Execute comprehensive dataset check"""
        from smartcash.ui.dataset.augmentation.utils.operation_handlers import handle_dataset_check
        handle_dataset_check(ui_components)
    
    def execute_cleanup_with_confirmation(button=None):
        """Execute cleanup dengan confirmation dialog"""
        from smartcash.ui.dataset.augmentation.utils.operation_handlers import handle_cleanup_with_confirmation
        handle_cleanup_with_confirmation(ui_components)
    
    # Bind operation handlers
    operation_handlers = {
        'augment_button': execute_augmentation_pipeline,
        'check_button': execute_dataset_check,
        'cleanup_button': execute_cleanup_with_confirmation
    }
    
    for button_key, handler in operation_handlers.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            button.on_click(handler)

def _setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config handlers dengan confirmation dialogs"""
    
    def save_config_with_validation(button=None):
        """Save config dengan validation dan confirmation"""
        from smartcash.ui.dataset.augmentation.utils.config_handlers import handle_save_config
        handle_save_config(ui_components)
    
    def reset_config_with_confirmation(button=None):
        """Reset config dengan confirmation dialog"""
        from smartcash.ui.dataset.augmentation.utils.config_handlers import handle_reset_config
        handle_reset_config(ui_components)
    
    # Bind config handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    
    if save_button and hasattr(save_button, 'on_click'):
        save_button.on_click(save_config_with_validation)
    if reset_button and hasattr(reset_button, 'on_click'):
        reset_button.on_click(reset_config_with_confirmation)