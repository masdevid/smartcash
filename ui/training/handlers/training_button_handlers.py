"""
File: smartcash/ui/training/handlers/training_button_handlers.py
Deskripsi: Button handlers untuk training controls dengan SRP approach
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager


# Global training state management
_training_state = {'active': False, 'stop_requested': False, 'model_ready': False}
get_state = lambda: _training_state
set_state = lambda **kwargs: _training_state.update(kwargs)


def setup_training_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup semua button handlers untuk training"""
    
    # Button state manager
    ui_components['button_state_manager'] = get_button_state_manager(ui_components)
    
    # Import specific handlers
    from smartcash.ui.training.handlers.start_training_handler import handle_start_training
    from smartcash.ui.training.handlers.stop_training_handler import handle_stop_training
    from smartcash.ui.training.handlers.reset_training_handler import handle_reset_training
    from smartcash.ui.training.handlers.validation_handler import handle_validate_model
    from smartcash.ui.training.handlers.cleanup_handler import handle_cleanup_gpu
    
    # Register button handlers
    button_handlers = {
        'start_button': lambda b: handle_start_training(ui_components, config),
        'stop_button': lambda b: handle_stop_training(ui_components),
        'reset_button': lambda b: handle_reset_training(ui_components),
        'validate_button': lambda b: handle_validate_model(ui_components, config),
        'cleanup_button': lambda b: handle_cleanup_gpu(ui_components)
    }
    
    # One-liner button registration
    [getattr(ui_components.get(btn), 'on_click', lambda x: None)(handler) 
     for btn, handler in button_handlers.items() if btn in ui_components]
    
    return ui_components


def update_button_states(ui_components: Dict[str, Any], training_active: bool):
    """Update button states berdasarkan training status"""
    button_states = {
        'start_button': training_active,
        'stop_button': not training_active,
        'reset_button': training_active,
        'validate_button': training_active,
        'cleanup_button': training_active
    }
    
    [setattr(ui_components.get(btn), 'disabled', disabled) 
     for btn, disabled in button_states.items() if btn in ui_components]


# One-liner utilities untuk button state management
enable_training_mode = lambda ui_components: update_button_states(ui_components, False)
enable_stopping_mode = lambda ui_components: update_button_states(ui_components, True)