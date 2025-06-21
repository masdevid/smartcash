"""
File: smartcash/ui/components/__init__.py
Deskripsi: Package untuk UI components dengan one-liner imports yang efisien
"""

from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
__all__ = [
    'create_status_panel', 
    'create_log_accordion', 
]