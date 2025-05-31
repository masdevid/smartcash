
"""
File: smartcash/ui/components/header.py
Deskripsi: Komponen header untuk UI dengan one-liner style
"""

from typing import Optional
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header as utils_create_header

def create_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    """Buat komponen header dengan one-liner style konsisten."""
    return utils_create_header(title, description, icon)

