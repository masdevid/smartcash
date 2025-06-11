"""
File: smartcash/ui/components/split_selector.py
Deskripsi: Komponen shared untuk pemilihan split dataset dengan one-liner style
"""

import ipywidgets as widgets
from typing import List
from smartcash.ui.utils.constants import ICONS

def create_split_selector(selected_value: str = 'All Splits', description: str = 'Process:', width: str = None,
                         icon: str = None, options: List[str] = None) -> widgets.RadioButtons:
    """Buat komponen UI untuk pemilihan split dataset dengan one-liner style."""
    split_options = options or ['All Splits', 'Train Only', 'Validation Only', 'Test Only']
    description = f"{ICONS[icon]} {description}" if icon and icon in ICONS else description
    layout_args = {'margin': '10px 0'}
    width and layout_args.update({'width': width})
    return widgets.RadioButtons(options=split_options, value=selected_value, description=description,
                               style={'description_width': 'initial'}, layout=widgets.Layout(**layout_args))
