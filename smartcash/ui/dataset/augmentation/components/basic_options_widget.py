"""
File: smartcash/ui/dataset/augmentation/components/basic_options_widget.py
Deskripsi: Widget UI murni untuk opsi dasar tanpa logika bisnis
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """
    Buat widget UI murni untuk opsi dasar augmentasi (tanpa logika bisnis).
    
    Returns:
        Dictionary berisi container dan mapping widget individual
    """
    # Jumlah variasi per gambar
    num_variations = widgets.IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Target count sebagai slider
    target_count = widgets.IntSlider(
        value=500,
        min=100,
        max=2000,
        step=100,
        description='Target Count:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Prefix output
    output_prefix = widgets.Text(
        value='aug_',
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Balance classes
    balance_classes = widgets.Checkbox(
        value=False,
        description='Balance Classes (Layer 1 & 2 only)',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Container
    container = widgets.VBox([
        num_variations,
        target_count,
        output_prefix,
        balance_classes
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'num_variations': num_variations,
            'target_count': target_count,
            'output_prefix': output_prefix,
            'balance_classes': balance_classes
        }
    }
