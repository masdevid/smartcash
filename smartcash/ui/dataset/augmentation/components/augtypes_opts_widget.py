"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Enhanced augmentation types widget dengan backend service integration
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_types_widget() -> Dict[str, Any]:
    """
    Create enhanced augmentation types dengan backend compatibility
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Enhanced jenis augmentasi dengan backend support - 65% width
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
    
    # Enhanced target split dengan backend guidance
    target_split = widgets.Dropdown(
        options=[
            ('🎯 Train - Dataset training (Backend Recommended)', 'train'),
            ('📊 Valid - Dataset validasi (Backend Support)', 'valid'),
            ('🧪 Test - Dataset testing (Not Recommended)', 'test')
        ],
        value='train',
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='100%'),
        style={'description_width': '140px'}
    )
    
    # Enhanced types information dengan backend details
    types_info = widgets.HTML(
        f"""
        <div style="padding: 12px; background-color:#2196f315; 
                    border-radius: 6px; margin: 10px 0; font-size: 12px;
                    border: 1px solid #2196f3;">
            <strong style="color:#2196f3">{ICONS.get('augmentation', '🔄')} Backend Augmentation Types:</strong><br>
            • <strong style="color:#2196f3">Combined</strong>: Research pipeline dengan optimized backend processing<br>
            • <strong style="color:#2196f3">Position</strong>: Geometric transforms dengan bbox preservation<br>
            • <strong style="color:#2196f3">Lighting</strong>: Photometric transforms untuk kondisi pencahayaan<br>
            • <strong style="color:#2196f3">Advanced Types</strong>: Backend support untuk geometric, color, noise transforms
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Enhanced split information dengan backend integration
    split_info = widgets.HTML(
        f"""
        <div style="padding: 12px; background-color:#2196f315; 
                    border-radius: 6px; margin: 10px 0; font-size: 12px;
                    border: 1px solid #2196f3;">
            <strong style="color:#2196f3">{ICONS.get('info', 'ℹ️')} Backend Split Management:</strong><br>
            • <strong style="color:#2196f3">train</strong>: Primary target dengan backend service integration<br>
            • <strong style="color:#2196f3">valid</strong>: Backend support untuk validation augmentation<br>
            • <strong style="color:#2196f3">test</strong>: Backend warning - tidak recommended untuk evaluation
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Types section (65% width)
    aug_types_section = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 10px 0;'>{ICONS.get('augmentation', '🔄')} Pilih Jenis Augmentasi (Backend Enhanced):</h6>"),
        augmentation_types,
        types_info
    ], layout=widgets.Layout(
        width='65%', 
        overflow='hidden', 
        display='flex', 
        flex_flow='column'
    ))
    
    # Split section (32% width)
    split_section = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 18px 0 10px 0;'>{ICONS.get('split', '📂')} Target Split Dataset:</h6>"),
        target_split,
        split_info
    ], layout=widgets.Layout(
        width='32%', 
        overflow='hidden', 
        display='flex', 
        flex_flow='column'
    ))
    
    # Main container dengan enhanced spacing
    container = widgets.HBox([
        aug_types_section,
        split_section
    ], layout=widgets.Layout(
        width='100%',
        padding='12px',
        margin='8px 0',
        display='flex',
        justify_content="space-between",
        align_items="stretch",
        gap='15px'
    ))
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types,
            'target_split': target_split
        },
        # Enhanced validation dengan backend compatibility
        'validation': {
            'required': ['augmentation_types', 'target_split'],
            'defaults': {
                'augmentation_types': ['combined'],
                'target_split': 'train'
            },
            'backend_compatible': True
        },
        # Extended options dengan backend support
        'options': {
            'augmentation_types': ['combined', 'position', 'lighting', 'geometric', 'color', 'noise'],
            'target_split': ['train', 'valid', 'test'],
            'backend_recommended': {
                'types': ['combined', 'position', 'lighting'],
                'split': 'train'
            }
        },
        # Backend integration metadata
        'backend_mapping': {
            'augmentation_types': 'augmentation.types',
            'target_split': 'augmentation.target_split'
        }
    }