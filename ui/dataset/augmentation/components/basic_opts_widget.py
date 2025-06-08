"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Basic options widget dengan validasi range dan UI responsive
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """
    Create basic options widget dengan range validation sesuai config
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jumlah variasi per gambar (1-10)
    num_variations = widgets.IntSlider(
        value=3, min=1, max=10, step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Target count dengan range realistis (100-2000)
    target_count = widgets.IntSlider(
        value=500, min=100, max=2000, step=100,
        description='Target Count:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Output prefix untuk naming
    output_prefix = widgets.Text(
        value='aug',
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Balance classes checkbox
    balance_classes = widgets.Checkbox(
        value=True,
        description='Balance Classes (Layer 1 & 2 only)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='5px 0')
    )
    
    # Info panel dengan guidance
    info_panel = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #4caf5015; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #4caf5040;" >
            <strong style="color: #2e7d32;">{ICONS.get('info', 'ℹ️')} Parameter Guidance:</strong><br>
            • <strong style="color: #2e7d32;">Variasi:</strong> 2-5 optimal untuk dataset research<br>
            • <strong style="color: #2e7d32;">Target Count:</strong> 500-1000 untuk training yang efektif<br>
            • <strong style="color: #2e7d32;">Balance Classes:</strong> Hanya Layer 1 & 2 (denominasi utama)
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Container dengan layout yang clean
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0;'>{ICONS.get('settings', '⚙️')} Opsi Dasar</h6>"),
        num_variations,
        target_count,
        output_prefix,
        balance_classes,
        info_panel
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'num_variations': num_variations,
            'target_count': target_count,
            'output_prefix': output_prefix,
            'balance_classes': balance_classes
        },
        # Validation info untuk form validation
        'validation': {
            'ranges': {
                'num_variations': (1, 10),
                'target_count': (100, 2000)
            },
            'required': ['num_variations', 'target_count', 'output_prefix']
        }
    }