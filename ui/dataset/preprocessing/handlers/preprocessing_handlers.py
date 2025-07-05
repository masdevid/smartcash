"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Main preprocessing handlers dengan centralized error handling dan SRP
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.dataset.preprocessing.handlers.operation import create_preprocessing_handler_manager

@handle_ui_errors(error_component_title="Preprocessing Handlers Setup Error", log_error=True)
def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup preprocessing handlers dengan centralized error handling dan SRP.
    
    Args:
        ui_components: Dictionary containing UI components
        config: Configuration dictionary
        env: Optional environment context
        
    Returns:
        Dictionary of UI components with handlers attached
    """
    # Create preprocessing handler manager
    manager = create_preprocessing_handler_manager(ui_components)
    
    # Setup handlers
    manager.setup_handlers(ui_components)
    
    # Store manager in UI components
    ui_components['preprocessing_handler_manager'] = manager
    
    # Log success
    manager.logger.info("✅ Preprocessing handlers berhasil disetup")
    
    return ui_components
