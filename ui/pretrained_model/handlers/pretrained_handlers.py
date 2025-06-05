"""
File: smartcash/ui/pretrained_model/handlers/pretrained_handlers.py
Deskripsi: Unified handlers yang terintegrasi dengan pretrained model services
"""

from typing import Dict, Any

def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan inline implementation"""
    
    # Import handlers yang diperlukan
    from smartcash.ui.pretrained_model.handlers.download_handler import setup_download_handler
    from smartcash.ui.pretrained_model.handlers.reset_handler import setup_reset_handler
    
    # Progress callback untuk model downloader
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'update'):
                ui_components['progress_tracker'].update('overall', progress, message)
            else:
                # Fallback ke metode lama
                ui_components.get('update_progress', lambda *a: None)('overall', progress, message)
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Status handler dengan integrasi status_panel
    def update_status(message: str, status_type: str = "info"):
        from smartcash.ui.components.status_panel import update_status_panel
        if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
            update_status_panel(ui_components['status_panel'], message, status_type)
    
    ui_components['update_status'] = update_status
    
    # Setup handlers
    setup_download_handler(ui_components, config)
    setup_reset_handler(ui_components)
    
    # Auto-check handler
    if ui_components.get('auto_check_enabled', False):
        from smartcash.ui.pretrained_model.handlers.check_handler import setup_check_handler
        setup_check_handler(ui_components, config)
    
    return ui_components

def _reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger dan clear all outputs"""
    [widget.clear_output(wait=True) for key in ['log_output', 'status'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
    ui_components.get('reset_all', lambda: None)()