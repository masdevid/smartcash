"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Fixed basic options dengan augmentation intensity slider
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """
    Create enhanced basic options widget dengan intensity slider
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jumlah variasi per gambar (1-10)
    num_variations = widgets.IntSlider(
        value=2, min=1, max=10, step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Target count dengan range optimal (100-2000)
    target_count = widgets.IntSlider(
        value=500, min=100, max=2000, step=50,
        description='Target Count:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # NEW: Augmentation Intensity slider
    intensity = widgets.FloatSlider(
        value=0.7, min=0.1, max=1.0, step=0.1,
        description='Intensitas:',
        continuous_update=False,
        readout=True, readout_format='.1f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Output prefix
    output_prefix = widgets.Text(
        value='aug',
        placeholder='Prefix untuk file hasil augmentasi (alphanumeric)',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Balance classes
    balance_classes = widgets.Checkbox(
        value=True,
        description='Balance Classes (Layer 1 & 2 optimal)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='8px 0')
    )
    
    # Enhanced info panel dengan intensity info
    info_panel = widgets.HTML(
        f"""
        <div style="padding: 10px; background-color: #4caf5015; 
                    border-radius: 6px; margin: 8px 0; font-size: 12px;
                    border: 1px solid #4caf5040;" >
            <strong style="color: #2e7d32;">{ICONS.get('info', 'ℹ️')} Parameter Guidance:</strong><br>
            • <strong style="color: #2e7d32;">Variasi:</strong> 2-5 optimal untuk research pipeline<br>
            • <strong style="color: #2e7d32;">Target Count:</strong> 500-1000 untuk training efektif<br>
            • <strong style="color: #2e7d32;">Intensitas:</strong> 0.7 optimal, 0.3-0.5 conservative, 0.8-1.0 aggressive<br>
            • <strong style="color: #2e7d32;">Balance Classes:</strong> Layer 1 & 2 (denominasi) optimal
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Container
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 8px 0;'>{ICONS.get('settings', '⚙️')} Opsi Dasar</h6>"),
        num_variations,
        target_count,
        intensity,  # NEW: Added intensity slider
        output_prefix,
        balance_classes,
        info_panel
    ], layout=widgets.Layout(padding='12px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'num_variations': num_variations,
            'target_count': target_count,
            'intensity': intensity,  # NEW: Added to widgets dict
            'output_prefix': output_prefix,
            'balance_classes': balance_classes
        },
        'validation': {
            'ranges': {
                'num_variations': (1, 10),
                'target_count': (100, 2000),
                'intensity': (0.1, 1.0)  # NEW: Added intensity range
            },
            'required': ['num_variations', 'target_count', 'intensity', 'output_prefix'],
            'backend_compatible': True
        },
        'backend_mapping': {
            'num_variations': 'augmentation.num_variations',
            'target_count': 'augmentation.target_count',
            'intensity': 'augmentation.intensity',  # NEW: Added intensity mapping
            'output_prefix': 'augmentation.output_prefix',
            'balance_classes': 'augmentation.balance_classes'
        }
    }