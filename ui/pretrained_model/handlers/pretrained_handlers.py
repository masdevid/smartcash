"""
File: smartcash/ui/pretrained_model/handlers/pretrained_handlers.py
Deskripsi: Unified handlers yang terintegrasi dengan pretrained model services
"""

from typing import Dict, Any
def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan integrasi pretrained model services"""
    
    # Setup handlers inline untuk menghindari circular imports
    setup_progress_handler(ui_components)
    setup_status_handler(ui_components)
    setup_download_handler(ui_components, config)
    setup_reset_handler(ui_components)
    
    return ui_components

# Helper functions
def _get_button_manager(ui_components: Dict[str, Any]):
    """Get button manager dengan fallback"""
    from smartcash.ui.utils.button_state_manager import get_button_state_manager
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = get_button_state_manager(ui_components)
    return ui_components['button_manager']

def _reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger dan clear all outputs"""
    for key in ['log_output', 'status']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)
    ui_components.get('reset_all', lambda: None)()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)

def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan integrasi pretrained model services"""
    
    # Setup progress callback untuk UI updates
    setup_progress_handler(ui_components)
    
    # Setup status handler untuk panel updates
    setup_status_handler(ui_components)
    
    # Setup download & sync handler
    setup_download_handler(ui_components, config)
    
    # Setup reset UI handler
    setup_reset_handler(ui_components)
    
    return ui_components