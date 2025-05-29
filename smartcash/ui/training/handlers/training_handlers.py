"""
File: smartcash/ui/training/handlers/training_handlers.py
Deskripsi: Main training handlers dengan simplified SRP approach
"""

from typing import Dict, Any
from smartcash.ui.training.handlers.training_button_handlers import setup_training_button_handlers
from smartcash.ui.training.utils.training_display_utils import update_training_info, prepare_model_background


def setup_all_training_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua training handlers dengan SRP approach"""
    
    # Setup button handlers
    ui_components = setup_training_button_handlers(ui_components, config)
    
    # Initialize training info display
    update_training_info(ui_components, config)
    
    # Prepare model dalam background
    prepare_model_background(ui_components, config)
    
    return ui_components


# Re-export handler functions untuk backward compatibility
def handle_start_training(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Re-export start training handler"""
    from smartcash.ui.training.handlers.start_training_handler import handle_start_training as start_handler
    return start_handler(ui_components, config)


def handle_stop_training(ui_components: Dict[str, Any]):
    """Re-export stop training handler"""
    from smartcash.ui.training.handlers.stop_training_handler import handle_stop_training as stop_handler
    return stop_handler(ui_components)


def handle_reset_training(ui_components: Dict[str, Any]):
    """Re-export reset training handler"""
    from smartcash.ui.training.handlers.reset_training_handler import handle_reset_training as reset_handler
    return reset_handler(ui_components)


def handle_validate_model(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Re-export validation handler"""
    from smartcash.ui.training.handlers.validation_handler import handle_validate_model as validate_handler
    return validate_handler(ui_components, config)


def handle_cleanup_gpu(ui_components: Dict[str, Any]):
    """Re-export cleanup handler"""
    from smartcash.ui.training.handlers.cleanup_handler import handle_cleanup_gpu as cleanup_handler
    return cleanup_handler(ui_components)