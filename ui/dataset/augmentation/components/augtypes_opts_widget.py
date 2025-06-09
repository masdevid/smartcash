"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Vertical layout untuk jenis augmentasi dan target split
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_types_widget() -> Dict[str, Any]:
    """
    Create vertical layout untuk augmentation types dan target split
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jenis augmentasi dengan backend support
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
        layout=widgets.Layout(width='100%', height='140px'),
        style={'description_width': '140px'}
    )
    
    # Target split dengan title yang diperbaiki
    target_split = widgets.Dropdown(
        options=[
            ('🎯 Train - Dataset training (Recommended)', 'train'),
            ('📊 Valid - Dataset validasi', 'valid'),
            ('🧪 Test - Dataset testing (Not Recommended)', 'test')
        ],
        value='train',
        description='Target Split Augmentasi:',  # FIXED: Changed title
        disabled=False,
        layout=widgets.Layout(width='100%'),
        style={'description_width': '160px'}
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
    
    # Split information
    split_info = widgets.HTML(
        f"""
        <div style="padding: 12px; background-color:#ff980315; 
                    border-radius: 6px; margin: 10px 0; font-size: 12px;
                    border: 1px solid #ff9803;">
            <strong style="color:#f57c00">{ICONS.get('info', 'ℹ️')} Target Split:</strong><br>
            • <strong style="color:#f57c00">train</strong>: Primary target untuk augmentasi<br>
            • <strong style="color:#f57c00">valid</strong>: Opsional untuk validation augmentation<br>
            • <strong style="color:#f57c00">test</strong>: Tidak recommended untuk evaluation
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # FIXED: Single column vertical layout
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 10px 0;'>{ICONS.get('augmentation', '🔄')} Jenis Augmentasi & Target Split</h6>"),
        augmentation_types,
        types_info,
        target_split,
        split_info
    ], layout=widgets.Layout(
        width='100%',
        padding='12px',
        margin='8px 0'
    ))
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types,
            'target_split': target_split
        },
        'validation': {
            'required': ['augmentation_types', 'target_split'],
            'defaults': {
                'augmentation_types': ['combined'],
                'target_split': 'train'
            },
            'backend_compatible': True
        },
        'options': {
            'augmentation_types': ['combined', 'position', 'lighting', 'geometric', 'color', 'noise'],
            'target_split': ['train', 'valid', 'test'],
            'backend_recommended': {
                'types': ['combined', 'position', 'lighting'],
                'split': 'train'
            }
        },
        'backend_mapping': {
            'augmentation_types': 'augmentation.types',
            'target_split': 'augmentation.target_split'
        }
    }