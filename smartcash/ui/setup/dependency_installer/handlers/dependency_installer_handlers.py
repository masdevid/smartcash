"""
File: smartcash/ui/setup/dependency_installer/handlers/dependency_installer_handlers.py
Deskripsi: Main handlers coordinator untuk dependency installer dengan SRP pattern
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.common.config.manager import get_config_manager

# Import SRP handlers
from smartcash.ui.setup.dependency_installer.handlers.installation_handler import setup_installation_handler
from smartcash.ui.setup.dependency_installer.handlers.analysis_handler import setup_analysis_handler
from smartcash.ui.setup.dependency_installer.handlers.status_check_handler import setup_status_check_handler
from smartcash.ui.setup.dependency_installer.handlers.config_handlers import setup_config_handlers

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers untuk dependency installer dengan pattern yang konsisten"""
    
    # Setup progress callback untuk semua handlers
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            progress_type = kwargs.get('type', 'overall')
            color = kwargs.get('color', None)
            ui_components.get('update_progress', lambda *a: None)(progress_type, progress, message, color)
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup button state manager
    ui_components['button_manager'] = get_button_state_manager(ui_components)
    
    # Setup individual handlers dengan SRP
    setup_installation_handler(ui_components, config)
    setup_analysis_handler(ui_components, config)
    setup_status_check_handler(ui_components, config)
    setup_config_handlers(ui_components, config)
    
    # Setup auto-analyze setelah UI render jika enabled
    if ui_components.get('auto_analyze_on_render', True):
        _setup_auto_analyze_on_render(ui_components)
    
    return ui_components

def _setup_auto_analyze_on_render(ui_components: Dict[str, Any]):
    """Setup auto-analyze packages setelah UI render"""
    import threading
    import time
    
    def auto_analyze_delayed():
        """Auto analyze dengan delay untuk ensure UI fully rendered"""
        time.sleep(1)  # Delay 1 detik untuk ensure UI ready
        
        logger = ui_components.get('logger')
        if logger:
            logger.info("🔍 Auto-analyzing packages after UI render...")
        
        # Trigger analysis handler
        if 'trigger_analysis' in ui_components and callable(ui_components['trigger_analysis']):
            ui_components['trigger_analysis']()
    
    # Check auto-analyze checkbox status
    auto_analyze_checkbox = ui_components.get('auto_analyze_checkbox')
    if auto_analyze_checkbox and auto_analyze_checkbox.value:
        # Run auto-analyze di background thread
        analysis_thread = threading.Thread(target=auto_analyze_delayed, daemon=True)
        analysis_thread.start()

def _clear_ui_outputs(ui_components: Dict[str, Any]):
    """Clear semua UI outputs - shared utility untuk handlers"""
    for key in ['log_output', 'status', 'confirmation_area']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel - shared utility untuk handlers"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)

def _get_button_manager(ui_components: Dict[str, Any]):
    """Get button manager instance - shared utility"""
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = get_button_state_manager(ui_components)
    return ui_components['button_manager']