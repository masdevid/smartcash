"""
File: smartcash/ui/dataset/augmentation/components/action_buttons.py
Deskripsi: Komponen tombol aksi untuk augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_action_buttons() -> Dict[str, Any]:
    """
    Buat komponen tombol aksi untuk augmentasi dataset.
    
    Returns:
        Dictionary berisi tombol dan containers
    """
    # Gunakan komponen standar untuk tombol aksi
    from smartcash.ui.components.action_buttons import create_action_buttons as create_std_buttons
    from smartcash.ui.components.action_buttons import create_visualization_buttons as create_std_vis_buttons
    
    # Tombol standar untuk aksi utama
    action_buttons = create_std_buttons(
        primary_label="Run Augmentation",
        primary_icon="random",
        cleanup_enabled=True
    )
    
    # Tombol untuk visualisasi
    visualization_buttons = create_std_vis_buttons()
    
    # Rename tombol sesuai konteks
    action_buttons['augment_button'] = action_buttons.pop('primary_button')
    
    # Kembalikan dictionary gabungan
    result = {
        'augment_button': action_buttons['augment_button'],
        'stop_button': action_buttons['stop_button'],
        'reset_button': action_buttons['reset_button'],
        'save_button': action_buttons['save_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'container': action_buttons['container'],
        'visualization_buttons': visualization_buttons['container'],
        'visualize_button': visualization_buttons['visualize_button'],
        'compare_button': visualization_buttons['compare_button'],
        'distribution_button': visualization_buttons['distribution_button']
    }
    
    # Set ke hidden by default
    visualization_buttons['container'].layout.display = 'none'
    action_buttons['cleanup_button'].layout.display = 'none'
    
    return result