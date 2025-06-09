"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Enhanced basic options widget dengan backend compatibility dan validation
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """
    Create enhanced basic options widget dengan backend integration
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jumlah variasi per gambar (1-10) - sesuai backend
    num_variations = widgets.IntSlider(
        value=2, min=1, max=10, step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Target count dengan range optimal untuk backend (100-2000)
    target_count = widgets.IntSlider(
        value=500, min=100, max=2000, step=50,
        description='Target Count:',
        continuous_update=False,
        readout=True, readout_format='d',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Output prefix untuk backend compatibility
    output_prefix = widgets.Text(
        value='aug',
        placeholder='Prefix untuk file hasil augmentasi (alphanumeric)',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Balance classes dengan enhanced description
    balance_classes = widgets.Checkbox(
        value=True,
        description='Balance Classes (Layer 1 & 2 optimal)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='8px 0')
    )
    
    # Enhanced info panel dengan backend integration info
    info_panel = widgets.HTML(
        f"""
        <div style="padding: 10px; background-color: #4caf5015; 
                    border-radius: 6px; margin: 8px 0; font-size: 12px;
                    border: 1px solid #4caf5040;" >
            <strong style="color: #2e7d32;">{ICONS.get('info', 'ℹ️')} Parameter Guidance (Backend Optimized):</strong><br>
            • <strong style="color: #2e7d32;">Variasi:</strong> 2-5 optimal untuk pipeline research dengan backend service<br>
            • <strong style="color: #2e7d32;">Target Count:</strong> 500-1000 untuk training efektif, backend support sampai 2000<br>
            • <strong style="color: #2e7d32;">Balance Classes:</strong> Layer 1 & 2 (denominasi) dengan algoritma backend<br>
            • <strong style="color: #2e7d32;">Backend:</strong> Service integration untuk progress tracking dan validation
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Container dengan enhanced layout
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 8px 0;'>{ICONS.get('settings', '⚙️')} Opsi Dasar - Backend Integration</h6>"),
        num_variations,
        target_count,
        output_prefix,
        balance_classes,
        info_panel
    ], layout=widgets.Layout(padding='12px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'num_variations': num_variations,
            'target_count': target_count,
            'output_prefix': output_prefix,
            'balance_classes': balance_classes
        },
        # Enhanced validation info untuk backend compatibility
        'validation': {
            'ranges': {
                'num_variations': (1, 10),
                'target_count': (100, 2000)
            },
            'required': ['num_variations', 'target_count', 'output_prefix'],
            'backend_compatible': True
        },
        # Backend integration metadata
        'backend_mapping': {
            'num_variations': 'augmentation.num_variations',
            'target_count': 'augmentation.target_count',
            'output_prefix': 'augmentation.output_prefix',
            'balance_classes': 'augmentation.balance_classes'
        }
    }