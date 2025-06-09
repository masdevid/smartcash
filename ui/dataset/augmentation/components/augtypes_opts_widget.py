"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Augmentation types widget tanpa target split (dipindah ke basic options)
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_types_widget() -> Dict[str, Any]:
    """Create augmentation types widget tanpa target split"""
    
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jenis augmentasi saja
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
        description='Jenis Augmentasi:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='160px'),
        style={'description_width': '140px'}
    )
    
    # Types information
    types_info = widgets.HTML(
        f"""
        <div style="padding: 12px; background-color:#2196f315; 
                    border-radius: 6px; margin: 10px 0; font-size: 12px;
                    border: 1px solid #2196f3;">
            <strong style="color:#2196f3">{ICONS.get('augmentation', '🔄')} Jenis Augmentasi:</strong><br>
            • <strong style="color:#2196f3">Combined</strong>: Research pipeline optimal<br>
            • <strong style="color:#2196f3">Position</strong>: Geometric transforms dengan bbox preservation<br>
            • <strong style="color:#2196f3">Lighting</strong>: Photometric transforms kondisi pencahayaan<br>
            • <strong style="color:#2196f3">Advanced</strong>: Geometric, color, noise transforms
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Container - hanya jenis augmentasi
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 10px 0;'>{ICONS.get('augmentation', '🔄')} Jenis Augmentasi</h6>"),
        augmentation_types,
        types_info
    ], layout=widgets.Layout(
        width='100%',
        padding='12px',
        margin='8px 0'
    ))
    
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