"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Augmentation types widget yang dioptimasi dengan styling terkonsolidasi
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.style_utils import (
    flex_layout, info_panel, create_info_content, section_header
)

def create_augmentation_types_widget() -> Dict[str, Any]:
    """Create compact augmentation types widget dengan styling terkonsolidasi"""
    
    # Augmentation types widget
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('🎯 Combined: Posisi + Pencahayaan (Research Pipeline)', 'combined'),
            ('📍 Position: Transformasi geometri (rotation, flip, scale)', 'position'),
            ('💡 Lighting: Variasi pencahayaan (brightness, contrast, HSV)', 'lighting'),
            ('🔄 Geometric: Transformasi lanjutan (perspective, shear)', 'geometric'),
            ('🎨 Color: Variasi warna dan saturasi', 'color'),
            ('📡 Noise: Gaussian noise dan motion blur', 'noise')
        ],
        value=['combined'],
        disabled=False,
        layout=widgets.Layout(width='100%', height='120px'),
        style={'description_width': '0'}
    )
    
    # Create info content menggunakan fungsi terkonsolidasi
    info_content = create_info_content([
        ('Jenis Augmentasi', ''),
        ('Combined', 'Research pipeline optimal'),
        ('Position', 'Geometric transforms + bbox preservation'),
        ('Lighting', 'Photometric transforms pencahayaan'),
        ('Advanced', 'Geometric, color, noise transforms')
    ], theme='types')
    
    # Create container dengan flex layout
    container = widgets.VBox([
        section_header('🔄 Jenis Augmentasi', theme='types'),
        augmentation_types,
        info_panel(info_content, theme='types')
    ])
    
    # Apply flex layout
    flex_layout(container)
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types
        },
        'validation': {
            'required': ['augmentation_types'],
            'defaults': {
                'augmentation_types': ['combined']
            },
            'backend_compatible': True
        },
        'options': {
            'augmentation_types': ['combined', 'position', 'lighting', 'geometric', 'color', 'noise'],
            'backend_recommended': {
                'types': ['combined', 'position', 'lighting']
            }
        },
        'backend_mapping': {
            'augmentation_types': 'augmentation.types'
        }
    }