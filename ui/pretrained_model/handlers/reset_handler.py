"""
File: smartcash/ui/pretrained_model/handlers/reset_handler.py
Deskripsi: Handler khusus untuk reset UI dengan SRP approach
"""

from typing import Dict, Any

def setup_reset_handler(ui_components: Dict[str, Any]):
    """Setup reset UI handler dengan comprehensive cleanup"""
    
    def execute_reset_ui(button=None):
        """Execute UI reset dengan complete cleanup"""
        logger = ui_components.get('logger')
        
        try:
            # Reset UI logger dan clear all outputs
            _reset_ui_logger(ui_components)
            
            # Reset progress tracking
            ui_components.get('reset_all', lambda: None)()
            
            # Reset status panel
            _update_status_panel(ui_components, "UI berhasil direset", "success")
            
            # Log reset action
            if logger:
                logger.info("🧹 UI berhasil direset - siap untuk operasi baru")
            
        except Exception as e:
            error_msg = f"Reset UI gagal: {str(e)}"
            if logger:
                logger.error(f"💥 {error_msg}")
            _update_status_panel(ui_components, error_msg, "error")
    
    ui_components['reset_ui_button'].on_click(execute_reset_ui)

def _reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger dan clear all outputs"""
    for key in ['log_output', 'status']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)