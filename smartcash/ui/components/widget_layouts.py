"""
File: smartcash/ui/components/widget_layouts.py
Deskripsi: Definisi layout standar untuk widgets UI dengan pendekatan factory
"""

import ipywidgets as widgets
from typing import Optional, Dict, Any, Union

from smartcash.ui.utils.constants import PADDINGS, MARGINS

def create_layout(
    width: str = '100%',
    height: Optional[str] = None,
    margin: str = MARGINS['medium'],
    padding: str = PADDINGS['medium'],
    border: Optional[str] = None,
    display: str = 'block',
    **kwargs
) -> widgets.Layout:
    """
    Factory untuk membuat layout dengan opsi yang fleksibel.
    
    Args:
        width: Lebar layout
        height: Tinggi layout (opsional)
        margin: Margin layout
        padding: Padding layout
        border: Border layout (opsional)
        display: Display type layout
        **kwargs: Parameter layout tambahan
        
    Returns:
        Layout widget dengan konfigurasi yang ditentukan
    """
    params = {
        'width': width,
        'margin': margin,
        'padding': padding,
        'display': display,
        **kwargs
    }
    
    if height:
        params['height'] = height
        
    if border:
        params['border'] = border
    
    return widgets.Layout(**params)

# Container Layouts
CONTAINER_LAYOUTS = {
    'main': create_layout(width='100%', padding=PADDINGS['medium']),
    'card': create_layout(
        border='1px solid #ddd',
        border_radius='4px',
        padding=PADDINGS['medium'],
        margin=MARGINS['medium']
    ),
    'section': create_layout(margin=f'{MARGINS["medium"]} 0'),
    'sidebar': create_layout(width='250px', padding=PADDINGS['small']),
}

# Content Layouts
CONTENT_LAYOUTS = {
    'output': create_layout(
        border='1px solid #ddd',
        min_height='100px',
        max_height='300px',
        margin=f'{MARGINS["medium"]} 0',
        overflow='auto'
    ),
    'status': create_layout(
        border='1px solid #ddd',
        min_height='50px',
        max_height='150px',
        margin=f'{MARGINS["small"]} 0',
        overflow='auto'
    ),
    'log': create_layout(
        border='1px solid #ddd',
        min_height='150px',
        max_height='300px',
        margin=f'{MARGINS["medium"]} 0',
        padding=PADDINGS['small'],
        overflow='auto',
        font_family='monospace'
    ),
}

# Input Layouts
INPUT_LAYOUTS = {
    'text': create_layout(width='60%', margin=f'{MARGINS["small"]} 0'),
    'textarea': create_layout(width='60%', height='150px', margin=f'{MARGINS["small"]} 0'),
    'dropdown': create_layout(width='50%', margin=f'{MARGINS["small"]} 0'),
    'slider': create_layout(width='70%', margin=f'{MARGINS["small"]} 0'),
    'checkbox': create_layout(margin=f'{MARGINS["small"]} 0'),
    'radio': create_layout(margin=f'{MARGINS["small"]} 0'),
}

# Button Layouts
BUTTON_LAYOUTS = {
    'standard': create_layout(margin=f'{MARGINS["medium"]} 0'),
    'small': create_layout(margin=MARGINS['small']),
    'hidden': create_layout(margin=f'{MARGINS["medium"]} 0', display='none'),
    'inline': create_layout(margin='0 5px 0 0', width='auto'),
}

# Group Layouts
GROUP_LAYOUTS = {
    'horizontal': create_layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        width='100%'
    ),
    'vertical': create_layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%'
    ),
    'grid': create_layout(
        display='grid',
        grid_gap='10px',
        grid_template_columns='repeat(auto-fit, minmax(250px, 1fr))',
        width='100%'
    ),
}

# Component Layouts
COMPONENT_LAYOUTS = {
    'tabs': create_layout(width='100%', margin=f'{MARGINS["medium"]} 0'),
    'accordion': create_layout(width='100%', margin=f'{MARGINS["medium"]} 0'),
    'progress': create_layout(width='100%', margin=f'{MARGINS["small"]} 0'),
    'divider': create_layout(
        height='1px',
        border='0',
        border_top='1px solid #eee',
        margin=f'{MARGINS["large"]} 0'
    ),
}

def create_divider() -> widgets.HTML:
    """Buat divider horizontal."""
    return widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>")

def create_spacing(height: str = '10px') -> widgets.HTML:
    """Buat elemen spacing untuk mengatur jarak antar komponen."""
    return widgets.HTML(f"<div style='height: {height};'></div>")

def create_grid_layout(
    nrows: int, 
    ncols: int, 
    layout_kwargs: Dict[str, Any] = None
) -> widgets.Layout:
    """
    Buat layout grid dengan jumlah baris dan kolom tertentu.
    
    Args:
        nrows: Jumlah baris
        ncols: Jumlah kolom
        layout_kwargs: Parameter layout tambahan
        
    Returns:
        Layout widget untuk grid
    """
    params = {
        'display': 'grid',
        'grid_template_rows': f'repeat({nrows}, auto)',
        'grid_template_columns': f'repeat({ncols}, 1fr)',
        'grid_gap': '10px',
        'width': '100%'
    }
    
    if layout_kwargs:
        params.update(layout_kwargs)
        
    return widgets.Layout(**params)

# Shortcut layouts for quick access
main_container = CONTAINER_LAYOUTS['main']
card_container = CONTAINER_LAYOUTS['card']
section_container = CONTAINER_LAYOUTS['section']
output_area = CONTENT_LAYOUTS['output']
status_area = CONTENT_LAYOUTS['status']
button = BUTTON_LAYOUTS['standard']
small_button = BUTTON_LAYOUTS['small']
hidden_button = BUTTON_LAYOUTS['hidden']
text_input = INPUT_LAYOUTS['text']
slider_input = INPUT_LAYOUTS['slider']
checkbox = INPUT_LAYOUTS['checkbox']
horizontal_group = GROUP_LAYOUTS['horizontal']
vertical_group = GROUP_LAYOUTS['vertical']