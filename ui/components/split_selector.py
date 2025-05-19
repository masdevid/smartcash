"""
File: smartcash/ui/components/split_selector.py
Deskripsi: Komponen shared untuk pemilihan split dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List
from smartcash.ui.utils.constants import ICONS

def create_split_selector(
    selected_value: str = 'All Splits',
    description: str = 'Process:',
    width: str = None,
    icon: str = None,
    options: List[str] = None
) -> widgets.RadioButtons:
    """
    Buat komponen UI untuk pemilihan split dataset.
    
    Args:
        selected_value: Nilai yang dipilih secara default
        description: Label deskripsi untuk selector
        width: Lebar komponen (opsional)
        icon: Ikon untuk ditampilkan di deskripsi (opsional)
        options: Opsi-opsi split kustom (opsional)
        
    Returns:
        Widget RadioButtons untuk pemilihan split
    """
    # Opsi-opsi split standar jika tidak ada opsi kustom
    split_options = options or ['All Splits', 'Train Only', 'Validation Only', 'Test Only']
    
    # Tambahkan ikon jika ada
    if icon and icon in ICONS:
        description = f"{ICONS[icon]} {description}"
    
    # Buat layout dengan width yang dapat dikustomisasi
    layout_args = {'margin': '10px 0'}
    if width:
        layout_args['width'] = width
    
    # Buat komponen RadioButtons
    split_selector = widgets.RadioButtons(
        options=split_options,
        value=selected_value,
        description=description,
        style={'description_width': 'initial'},
        layout=widgets.Layout(**layout_args)
    )
    
    return split_selector
