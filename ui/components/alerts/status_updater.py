"""
File: smartcash/ui/components/alerts/status_updater.py
Deskripsi: Fungsi untuk update status panel UI
"""
from typing import Dict, Any

def update_status(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """
    Update status panel dengan alert
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan status
        status_type: Tipe status ('info', 'success', 'warning', 'error')
    """
    if 'status_panel' in ui_components:
        from smartcash.ui.components import update_status_panel
        update_status_panel(ui_components['status_panel'], message, status_type)
