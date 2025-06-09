"""
File: smartcash/ui/dataset/augmentation/components/normalization_opts_widget.py
Deskripsi: Normalization options widget yang dioptimasi dengan styling terkonsolidasi
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_normalization_options_widget() -> Dict[str, Any]:
    """Create normalization options dengan styling terkonsolidasi"""
    
    from smartcash.ui.dataset.augmentation.utils.style_utils import (
        style_widget, flex_layout, info_panel, create_info_content, section_header
    )
    
    # Create widgets dengan consistent styling
    widgets_dict = {
        'norm_method': widgets.Dropdown(
            options=[
                ('MinMax (0-1): YOLOv5 compatible', 'minmax'),
                ('Standard (z-score): Statistical normalization', 'standard'),
                ('ImageNet: Transfer learning preset', 'imagenet'),
                ('None: Raw values (0-255)', 'none')
            ],
            value='minmax', description='Norm Method:', disabled=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='95%')
        ),
        'denormalize': widgets.Checkbox(
            value=False,
            description='Denormalize setelah preprocessing (save as uint8)',
            indent=False,
            layout=widgets.Layout(width='auto', margin='6px 0')
        )
    }
    
    # Create info content menggunakan fungsi terkonsolidasi
    info_content = create_info_content([
        ('Normalization Backend', ''),
        ('MinMax', 'OpenCV + NumPy [0.0, 1.0]'),
        ('Standard', 'Scikit-learn compatible'),
        ('ImageNet', 'Torchvision preset'),
        ('Target', '640x640 fixed untuk YOLO')
    ], theme='normalization')
    
    # Create container dengan flex layout
    container = widgets.VBox([
        section_header('📊 Normalisasi Augmentasi', theme='normalization'),
        *widgets_dict.values(),
        info_panel(info_content, theme='normalization')
    ])
    
    # Apply flex layout
    flex_layout(container)
    
    return {
        'container': container,
        'widgets': widgets_dict,
        'validation': {
            'ranges': {},
            'required': ['norm_method'],
            'defaults': {
                'norm_method': 'minmax',
                'denormalize': False
            }
        },
        'backend_mapping': {
            'preprocessing': {
                'normalization_method': 'norm_method',
                'denormalize_output': 'denormalize',
                'target_size': [640, 640]
            }
        }
    }