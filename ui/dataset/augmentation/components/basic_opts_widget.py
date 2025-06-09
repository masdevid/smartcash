"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Compact info panel dengan purple colors dan reduced spacing
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """Create basic options widget dengan compact styling"""
    
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Widgets dengan flex layout
    num_variations = widgets.IntSlider(
        value=2, min=1, max=10, step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    target_count = widgets.IntSlider(
        value=500, min=100, max=2000, step=50,
        description='Target Count:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    intensity = widgets.FloatSlider(
        value=0.7, min=0.1, max=1.0, step=0.1,
        description='Intensitas:',
        continuous_update=False,
        readout=True, readout_format='.1f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    target_split = widgets.Dropdown(
        options=[
            ('🎯 Train - Dataset training (Recommended)', 'train'),
            ('📊 Valid - Dataset validasi', 'valid'),
            ('🧪 Test - Dataset testing (Not Recommended)', 'test')
        ],
        value='train',
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    output_prefix = widgets.Text(
        value='aug',
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    balance_classes = widgets.Checkbox(
        value=True,
        description='Balance Classes (Layer 1 & 2 optimal)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='6px 0')
    )
    
    # FIXED: Compact info panel dengan purple colors
    info_panel = widgets.HTML(
        f"""
        <div style="padding: 6px 8px; background-color: #4caf5015; 
                    border-radius: 4px; margin: 6px 0; font-size: 10px;
                    border: 1px solid #4caf5040; line-height: 1.3;">
            <strong style="color: #2e7d32;">{ICONS.get('info', 'ℹ️')} Parameter Guidance:</strong><br>
            • <strong style="color: #2e7d32;">Variasi:</strong> 2-5 optimal untuk research<br>
            • <strong style="color: #2e7d32;">Target Count:</strong> 500-1000 efektif<br>
            • <strong style="color: #2e7d32;">Intensitas:</strong> 0.7 optimal, 0.3-0.5 conservative<br>
            • <strong style="color: #2e7d32;">Target Split:</strong> Train primary untuk augmentasi
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Container dengan flexbox
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 6px 0;'>{ICONS.get('settings', '⚙️')} Opsi Dasar</h6>"),
        num_variations,
        target_count,
        intensity,
        target_split,
        output_prefix,
        balance_classes,
        info_panel
    ], layout=widgets.Layout(
        padding='10px', 
        width='100%',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        gap='4px'
    ))
    
    return {
        'container': container,
        'widgets': {
            'num_variations': num_variations,
            'target_count': target_count,
            'intensity': intensity,
            'target_split': target_split,
            'output_prefix': output_prefix,
            'balance_classes': balance_classes
        },
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