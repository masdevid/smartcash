"""
File: smartcash/ui/training/utils/training_status_utils.py
Deskripsi: Utilities untuk training status management
"""

from typing import Dict, Any


def update_training_status(ui_components: Dict[str, Any], message: str, status_type: str):
    """Update status panel dengan message dan type"""
    from smartcash.ui.components.status_panel import update_status_panel
    
    status_panel = ui_components.get('status_panel')
    status_panel and update_status_panel(status_panel, message, status_type)