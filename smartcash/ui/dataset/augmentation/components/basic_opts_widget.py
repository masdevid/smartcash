"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Basic options widget yang dioptimasi dengan fungsi styling terkonsolidasi
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """Create basic options dengan styling terkonsolidasi"""
    
    from smartcash.ui.dataset.augmentation.utils.style_utils import (
        style_widget, flex_layout, info_panel, create_info_content
    )
    
    # Create widgets dengan consistent styling
    widgets_dict = {
        'num_variations': style_widget(widgets.IntSlider(
            value=2, min=1, max=10, step=1, description='Jumlah Variasi:',
            continuous_update=False, readout=True, readout_format='d'
        )),
        'target_count': style_widget(widgets.IntSlider(
            value=500, min=100, max=2000, step=50, description='Target Count:',
            continuous_update=False, readout=True, readout_format='d'
        )),
        'intensity': style_widget(widgets.FloatSlider(
            value=0.7, min=0.1, max=1.0, step=0.1, description='Intensitas:',
            continuous_update=False, readout=True, readout_format='.1f'
        )),
        'target_split': style_widget(widgets.Dropdown(
            options=[
                ('🎯 Train - Dataset training (Recommended)', 'train'),
                ('📊 Valid - Dataset validasi', 'valid'),
                ('🧪 Test - Dataset testing (Not Recommended)', 'test')
            ],
            value='train', description='Target Split:', disabled=False
        )),
        'output_prefix': style_widget(widgets.Text(
            value='aug', placeholder='Prefix untuk file hasil augmentasi',
            description='Output Prefix:', disabled=False
        )),
        'balance_classes': widgets.Checkbox(
            value=True, description='Balance Classes (Layer 1 & 2 optimal)',
            indent=False, layout=widgets.Layout(width='auto', margin='6px 0')
        )
    }
    
    # Create info content menggunakan fungsi terkonsolidasi
    info_content = create_info_content([
        ('Parameter Guidance', ''),
        ('Variasi', '2-5 optimal untuk research'),
        ('Target Count', '500-1000 efektif'),
        ('Intensitas', '0.7 optimal, 0.3-0.5 conservative'),
        ('Target Split', 'Train primary untuk augmentasi')
    ], theme='basic')
    
    # Create container dengan flex layout
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #4caf50; margin: 6px 0;'>⚙️ Opsi Dasar</h6>"),
        *widgets_dict.values(),
        info_panel(info_content, theme='basic')
    ])
    
    # Apply flex layout
    flex_layout(container)
    
    return {
        'container': container,
        'widgets': widgets_dict,
        'validation': {
            'ranges': {
                'num_variations': (1, 10),
                'target_count': (100, 2000),
                'intensity': (0.1, 1.0)
            },
            'required': ['num_variations', 'target_count', 'intensity', 'target_split', 'output_prefix'],
            'backend_compatible': True
        },
        'backend_mapping': {
            'num_variations': 'augmentation.num_variations',
            'target_count': 'augmentation.target_count',
            'intensity': 'augmentation.intensity',
            'target_split': 'augmentation.target_split',
            'output_prefix': 'augmentation.output_prefix',
            'balance_classes': 'augmentation.balance_classes'
        }
    }