
"""
File: smartcash/ui/dataset/shared/status_panel.py
Deskripsi: Utilitas shared untuk mengelola panel status pada modul dataset
"""

from typing import Dict, Any

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update status panel dengan pesan dan jenis status.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan status
    """
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    if 'status_panel' in ui_components:
        ui_components['status_panel'].value = create_info_alert(message, status_type).value