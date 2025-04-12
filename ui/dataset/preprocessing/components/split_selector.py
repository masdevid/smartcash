"""
File: smartcash/ui/dataset/preprocessing/components/split_selector.py
Deskripsi: Komponen pemilihan split untuk preprocessing dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List

def create_split_selector(selected_value: str = 'All Splits') -> widgets.RadioButtons:
    """
    Buat komponen UI untuk pemilihan split dataset.
    
    Args:
        selected_value: Nilai yang dipilih secara default
        
    Returns:
        Widget RadioButtons untuk pemilihan split
    """
    # Opsi-opsi split standar
    split_options = ['All Splits', 'Train Only', 'Validation Only', 'Test Only']
    
    # Buat komponen RadioButtons
    split_selector = widgets.RadioButtons(
        options=split_options,
        value=selected_value,
        description='Process:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='10px 0')
    )
    
    return split_selector