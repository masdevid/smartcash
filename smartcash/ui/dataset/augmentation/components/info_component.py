"""
File: smartcash/ui/dataset/augmentation/components/info_component.py
Deskripsi: Komponen informasi bantuan untuk augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_info_component() -> widgets.Widget:
    """
    Buat komponen informasi bantuan untuk augmentasi dataset.
    
    Returns:
        Widget informasi
    """
    # Gunakan info box standar
    from smartcash.ui.info_boxes.augmentation_info import get_augmentation_info
    help_panel = get_augmentation_info(open_by_default=False)
    
    return help_panel