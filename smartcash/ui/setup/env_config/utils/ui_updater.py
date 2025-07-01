"""
File: smartcash/ui/setup/env_config/utils/ui_updater.py
Deskripsi: Utilities untuk update UI components
"""

import ipywidgets as widgets
from typing import Dict, Any


def update_status_panel(status_widget: widgets.HTML, message: str, status_type: str = "info") -> None:
    """📋 Update status panel dengan styling"""
    colors = {
        'info': '#2196f3',
        'success': '#4caf50', 
        'warning': '#ff9800',
        'danger': '#f44336'
    }
    
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'danger': '❌'
    }
    
    color = colors.get(status_type, '#2196f3')
    icon = icons.get(status_type, 'ℹ️')
    
    status_widget.value = f"""
    <div style="background: {color}15; border-left: 4px solid {color}; padding: 10px; border-radius: 4px;">
        <span style="color: {color}; font-weight: bold;">{icon} {message}</span>
    </div>
    """