"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Basic options widget dengan cleanup options integration dan responsive styling
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.style_utils import (
    style_widget, flex_layout, info_panel, create_info_content
)

def create_basic_options_widget() -> Dict[str, Any]:
    """Create basic options dengan cleanup integration dan responsive styling"""
    
    # Create widgets dengan overflow-safe styling
    widgets_dict = {
        'num_variations': widgets.IntSlider(
            value=2, min=1, max=10, step=1, description='Jumlah Variasi:',
            continuous_update=False, readout=True, readout_format='d',
            style={'description_width': '110px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'target_count': widgets.IntSlider(
            value=500, min=100, max=2000, step=50, description='Target Count:',
            continuous_update=False, readout=True, readout_format='d',
            style={'description_width': '110px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'intensity': widgets.FloatSlider(
            value=0.7, min=0.1, max=1.0, step=0.1, description='Intensitas:',
            continuous_update=False, readout=True, readout_format='.1f',
            style={'description_width': '110px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'target_split': widgets.Dropdown(
            options=[
                ('🎯 Train - Dataset training (Recommended)', 'train'),
                ('📊 Valid - Dataset validasi', 'valid'),
                ('🧪 Test - Dataset testing (Not Recommended)', 'test')
            ],
            value='train', description='Target Split:', disabled=False,
            style={'description_width': '110px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        # CHANGED: cleanup_target menggantikan output_prefix
        'cleanup_target': widgets.Dropdown(
            options=[
                ('🧹 Augmented - Hapus file augmented saja', 'augmented'),
                ('🖼️ Samples - Hapus sample preview saja', 'samples'),
                ('🗑️ Both - Hapus augmented + samples', 'both')
            ],
            value='both', description='Cleanup Target:', disabled=False,
            style={'description_width': '110px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'balance_classes': widgets.Checkbox(
            value=True, description='Balance Classes (Layer 1 & 2 optimal)',
            indent=False, layout=widgets.Layout(width='auto', margin='6px 0')
        )
    }
    
    # Create info content dengan cleanup guidance
    info_content = create_info_content([
        ('Parameter Guidance', ''),
        ('Variasi', '2-5 optimal untuk research'),
        ('Target Count', '500-1000 efektif'),
        ('Intensitas', '0.7 optimal, 0.3-0.5 conservative'),
        ('Cleanup', 'Both = comprehensive cleanup')
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
            'required': ['num_variations', 'target_count', 'intensity', 'target_split', 'cleanup_target'],
            'backend_compatible': True
        },
        'backend_mapping': {
            'num_variations': 'augmentation.num_variations',
            'target_count': 'augmentation.target_count',
            'intensity': 'augmentation.intensity',
            'target_split': 'augmentation.target_split',
            'cleanup_target': 'cleanup.default_target',
            'balance_classes': 'augmentation.balance_classes'
        }
    }