"""
File: smartcash/ui/pretrained_model/handlers/status_handler.py
Deskripsi: Handler khusus untuk status panel updates dengan SRP approach
"""

from typing import Dict, Any

def setup_status_handler(ui_components: Dict[str, Any]):
    """Setup status handler untuk consistent panel updates"""
    
    def update_status(message: str, status_type: str = "info"):
        """Update status panel dengan formatting yang konsisten"""
        from smartcash.ui.components.status_panel import update_status_panel
        if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
            update_status_panel(ui_components['status_panel'], message, status_type)
    
    ui_components['update_status'] = update_status