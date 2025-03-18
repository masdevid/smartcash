"""
File: smartcash/ui/components/headers.py
Deskripsi: Komponen header dan section title yang menggunakan ui_helpers untuk konsistensi
"""
from typing import Optional
import ipywidgets as widgets

# Import fungsi dari ui_helpers untuk konsistensi
from smartcash.ui.utils.ui_helpers import (
    create_header as ui_helpers_create_header,
    create_section_title as ui_helpers_create_section_title,
    create_tab_view as ui_helpers_create_tab_view
)

def create_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat komponen header dengan style konsisten.
    
    Args:
        title: Judul header
        description: Deskripsi opsional
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi header
    """
    return ui_helpers_create_header(title, description, icon)

def create_component_header(title, description="", icon="🔧"):
    """Alias untuk create_header dengan parameter yang dibalik untuk backward compatibility."""
    return create_header(title, description, icon)

def create_section_title(title: str, icon: Optional[str] = "") -> widgets.HTML:
    """
    Buat judul section dengan style konsisten.
    
    Args:
        title: Judul section
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi judul section
    """
    return ui_helpers_create_section_title(title, icon)

# Alias untuk backward compatibility
create_section_header = create_section_title