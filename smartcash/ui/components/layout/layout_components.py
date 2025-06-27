"""
File: smartcash/ui/components/layout/layout_components.py
Deskripsi: Komponen layout yang dapat digunakan kembali untuk UI
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union, List, Tuple
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# Layout standar yang sering digunakan
LAYOUTS = {
    'header': widgets.Layout(margin='0 0 15px 0'),
    'section': widgets.Layout(margin='15px 0 10px 0'),
    'container': widgets.Layout(width='100%', padding='10px'),
    'output': widgets.Layout(width='100%', border='1px solid #ddd', min_height='100px', 
                           max_height='300px', margin='10px 0', overflow='auto'),
    'button': widgets.Layout(margin='10px 0'),
    'button_small': widgets.Layout(margin='5px'),
    'button_hidden': widgets.Layout(margin='10px 0', display='none'),
    'text_input': widgets.Layout(width='60%', margin='10px 0'),
    'text_area': widgets.Layout(width='60%', height='150px', margin='10px 0'),
    'selection': widgets.Layout(margin='10px 0'),
    'hbox': widgets.Layout(display='flex', flex_flow='row wrap', align_items='center', width='100%'),
    'vbox': widgets.Layout(display='flex', flex_flow='column', align_items='stretch', width='100%'),
    'divider': widgets.Layout(height='1px', border='0', border_top='1px solid #eee', margin='15px 0'),
    'card': widgets.Layout(border='1px solid #ddd', border_radius='4px', padding='15px', margin='10px 0', width='100%'),
    'tabs': widgets.Layout(width='100%', margin='10px 0'),
    'accordion': widgets.Layout(width='100%', margin='10px 0'),
    
    # Responsive layouts
    'responsive_container': widgets.Layout(width='100%', max_width='100%', padding='10px', overflow='hidden'),
    'responsive_button': widgets.Layout(width='auto', max_width='150px', height='32px', margin='2px', overflow='hidden'),
    'responsive_dropdown': widgets.Layout(width='100%', max_width='100%', margin='3px 0', overflow='hidden'),
    'two_column_left': widgets.Layout(width='47%', margin='0', padding='4px', overflow='hidden'),
    'two_column_right': widgets.Layout(width='47%', margin='0', padding='4px', overflow='hidden'),
    'no_scroll': widgets.Layout(overflow='hidden', max_width='100%')
}

def create_element(element_type: str, content: Union[str, list] = "", **kwargs) -> widgets.Widget:
    """
    Fungsi terpadu untuk membuat berbagai elemen UI dengan fitur responsif.
    
    Args:
        element_type: Jenis elemen ('divider', 'container', 'two_column', 'slider', 'dropdown', 'checkbox', 'text_input', 'log_slider')
        content: Konten untuk elemen (string untuk divider, list untuk container)
        **kwargs: Parameter tambahan spesifik untuk tiap tipe elemen
        
    Returns:
        Widget yang sesuai dengan element_type
    """
    if element_type == 'divider':
        margin, color, height = kwargs.get('margin', '15px 0'), kwargs.get('color', '#eee'), kwargs.get('height', '1px')
        return widgets.HTML(f"<hr style='margin: {margin}; border: 0; border-top: {height} solid {color};'>", 
                          layout=LAYOUTS['divider'])
    
    elif element_type == 'container':
        children, container_type = content, kwargs.get('container_type', 'vbox')
        layout = LAYOUTS[container_type].copy()
        layout.update({k: v for k, v in kwargs.items() if k in layout.keys()})
        
        if container_type == 'hbox':
            return widgets.HBox(children, layout=layout)
        else:  # vbox
            return widgets.VBox(children, layout=layout)
    
    elif element_type == 'slider':
        value = kwargs.get('value', 0)
        min_val = kwargs.get('min_val', 0)
        max_val = kwargs.get('max_val', 100)
        step = kwargs.get('step', 1)
        description = kwargs.get('description', '')
        tooltip = kwargs.get('tooltip', '')
        style = kwargs.get('style', {'description_width': '140px', 'handle_color': '#4CAF50'})
        
        if isinstance(value, int) and isinstance(step, int):
            return widgets.IntSlider(
                value=value, min=min_val, max=max_val, step=step,
                description=description, tooltip=tooltip, style=style,
                layout=LAYOUTS['responsive_dropdown'].copy()
            )
        else:
            return widgets.FloatSlider(
                value=value, min=min_val, max=max_val, step=step,
                description=description, tooltip=tooltip, style=style,
                layout=LAYOUTS['responsive_dropdown'].copy()
            )
    
    elif element_type == 'dropdown':
        return widgets.Dropdown(
            value=kwargs.get('value'),
            options=kwargs.get('options', []),
            description=kwargs.get('description', ''),
            tooltip=kwargs.get('tooltip', ''),
            style={'description_width': '140px'},
            layout=widgets.Layout(
                width='100%',
                max_width='100%',
                margin='3px 0',
                overflow='hidden'
            )
        )
    
    elif element_type == 'checkbox':
        return widgets.Checkbox(
            value=kwargs.get('value', False),
            description=kwargs.get('description', ''),
            tooltip=kwargs.get('tooltip', ''),
            disabled=kwargs.get('disabled', False),
            style={'description_width': '200px'},
            layout=widgets.Layout(
                width='100%',
                max_width='100%',
                margin='3px 0',
                overflow='hidden'
            )
        )
    
    elif element_type == 'text_input':
        return widgets.Text(
            value=kwargs.get('value', ''),
            description=kwargs.get('description', ''),
            placeholder=kwargs.get('placeholder', ''),
            tooltip=kwargs.get('tooltip', ''),
            style={'description_width': '140px'},
            layout=widgets.Layout(
                width='100%',
                max_width='100%',
                margin='3px 0',
                overflow='hidden'
            )
        )
    
    elif element_type == 'log_slider':
        return widgets.FloatLogSlider(
            value=kwargs.get('value', 0.001),
            min=kwargs.get('min_val', -5),
            max=kwargs.get('max_val', 0),
            step=kwargs.get('step', 0.1),
            description=kwargs.get('description', ''),
            tooltip=kwargs.get('tooltip', ''),
            style={'description_width': '140px'},
            layout=LAYOUTS['responsive_dropdown'].copy()
        )
    
    elif element_type == 'two_column':
        left_content, right_content = content
        left_width, right_width = kwargs.get('left_width', '48%'), kwargs.get('right_width', '48%')
        vertical_align = kwargs.get('vertical_align', 'flex-start')
        
        # Create new layout objects instead of copying
        left_layout = widgets.Layout(**{
            **LAYOUTS['two_column_left'],
            'width': left_width
        })
        right_layout = widgets.Layout(**{
            **LAYOUTS['two_column_right'],
            'width': right_width
        })
        
        left_wrapper = widgets.VBox([left_content], layout=left_layout)
        right_wrapper = widgets.VBox([right_content], layout=right_layout)
        
        return widgets.HBox(
            [left_wrapper, right_wrapper], 
            layout=widgets.Layout(
                width='100%', 
                max_width='100%', 
                justify_content='space-between', 
                align_items=vertical_align, 
                margin='0', 
                padding='0', 
                overflow='hidden'
            )
        )
    
    else:
        raise ValueError(f"Tipe elemen tidak didukung: {element_type}")

# Fungsi bantuan responsif
def get_responsive_config(widget_type: str, **kwargs) -> Dict[str, Any]:
    """Dapatkan konfigurasi responsif untuk berbagai tipe widget"""
    configs = {
        'button': {'layout': LAYOUTS['responsive_button'].copy()},
        'dropdown': {
            'layout': LAYOUTS['responsive_dropdown'].copy(),
            'style': {'description_width': kwargs.get('description_width', '80px')}
        },
        'input': {'layout': LAYOUTS['responsive_dropdown'].copy()}
    }
    config = configs.get(widget_type, {'layout': LAYOUTS['no_scroll'].copy()})
    
    # Terapkan override yang diberikan
    if 'layout' in config:
        config['layout'].update({k: v for k, v in kwargs.items() if k in config['layout'].keys()})
    
    return config

# Fungsi bantuan satu baris
create_divider = lambda margin='15px 0', color='#eee', height='1px': create_element('divider', margin=margin, color=color, height=height)
create_responsive_container = lambda children, container_type='vbox', **kwargs: create_element('container', children, container_type=container_type, **kwargs)

def create_responsive_two_column(left_content, right_content, **kwargs):
    """Create a responsive two-column layout.
    
    Args:
        left_content: Widget or component for the left column
        right_content: Widget or component for the right column
        **kwargs: Additional layout parameters
        
    Returns:
        HBox containing the two columns
    """
    left_width = kwargs.pop('left_width', '48%')
    right_width = kwargs.pop('right_width', '48%')
    vertical_align = kwargs.pop('vertical_align', 'flex-start')
    
    # Create new layout objects
    left_layout = widgets.Layout(
        width=left_width,
        margin='0',
        padding='4px',
        overflow='hidden'
    )
    
    right_layout = widgets.Layout(
        width=right_width,
        margin='0',
        padding='4px',
        overflow='hidden'
    )
    
    left_wrapper = widgets.VBox([left_content], layout=left_layout)
    right_wrapper = widgets.VBox([right_content], layout=right_layout)
    
    return widgets.HBox(
        [left_wrapper, right_wrapper],
        layout=widgets.Layout(
            width='100%',
            max_width='100%',
            justify_content='space-between',
            align_items=vertical_align,
            margin='0',
            padding='0',
            overflow='hidden',
            **kwargs.get('layout', {})
        )
    )

get_responsive_button_layout = lambda width='auto', max_width='150px': get_responsive_config('button', width=width, max_width=max_width)['layout']
