"""
File: smartcash/ui/utils/layout_utils.py
Deskripsi: Consolidated layout utilities with backward compatibility and responsive features
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union

# =============================================================================
# CONSOLIDATED LAYOUT CONSTANTS
# =============================================================================

# All-in-one layout dictionary combining standard and responsive layouts
LAYOUTS = {
    # Standard layouts (backward compatibility)
    'header': widgets.Layout(margin='0 0 15px 0'),
    'section': widgets.Layout(margin='15px 0 10px 0'),
    'container': widgets.Layout(width='100%', padding='10px'),
    'output': widgets.Layout(width='100%', border='1px solid #ddd', min_height='100px', max_height='300px', margin='10px 0', overflow='auto'),
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
    
    # Responsive layouts (enhanced versions)
    'responsive_container': widgets.Layout(width='100%', max_width='100%', padding='10px', overflow='hidden'),
    'responsive_button': widgets.Layout(width='auto', max_width='150px', height='32px', margin='2px', overflow='hidden'),
    'responsive_dropdown': widgets.Layout(width='100%', max_width='100%', margin='3px 0', overflow='hidden'),
    'two_column_left': widgets.Layout(width='47%', margin='0', padding='4px', overflow='hidden'),
    'two_column_right': widgets.Layout(width='47%', margin='0', padding='4px', overflow='hidden'),
    'no_scroll': widgets.Layout(overflow='hidden', max_width='100%')
}

# Backward compatibility aliases
MAIN_CONTAINER = LAYOUTS['container']
OUTPUT_WIDGET = LAYOUTS['output']
BUTTON = LAYOUTS['button']
HIDDEN_BUTTON = LAYOUTS['button_hidden']
TEXT_INPUT = LAYOUTS['text_input']
TEXT_AREA = LAYOUTS['text_area']
SELECTION = LAYOUTS['selection']
HORIZONTAL_GROUP = LAYOUTS['hbox']
VERTICAL_GROUP = LAYOUTS['vbox']
DIVIDER = LAYOUTS['divider']
CARD = LAYOUTS['card']
TABS = LAYOUTS['tabs']
ACCORDION = LAYOUTS['accordion']
STANDARD_LAYOUTS = {k: v for k, v in LAYOUTS.items() if not k.startswith('responsive_') and not k.startswith('two_column_')}
RESPONSIVE_LAYOUTS = {k: v for k, v in LAYOUTS.items() if k.startswith('responsive_') or k.startswith('two_column_') or k == 'no_scroll'}

# =============================================================================
# CONSOLIDATED UTILITY FUNCTIONS
# =============================================================================

def create_element(element_type: str, content: Union[str, list] = "", **kwargs) -> widgets.Widget:
    """
    Unified function to create various UI elements with responsive features.
    
    Args:
        element_type: Type of element ('divider', 'header', 'container', 'two_column')
        content: Content for the element (string for divider/header, list for containers)
        **kwargs: Additional parameters specific to element type
        
    Returns:
        Appropriate widget based on element_type
    """
    if element_type == 'divider':
        margin, color, height = kwargs.get('margin', '15px 0'), kwargs.get('color', '#eee'), kwargs.get('height', '1px')
        return widgets.HTML(f"<hr style='margin: {margin}; border: 0; border-top: {height} solid {color};'>", layout=widgets.Layout(width='100%', margin='0', padding='0'))
    
    elif element_type == 'header':
        title, icon, color, font_size, margin = content, kwargs.get('icon', '📋'), kwargs.get('color', '#333'), kwargs.get('font_size', '16px'), kwargs.get('margin', '15px 0 8px 0')
        return widgets.HTML(f"<h4 style='color: {color}; margin: {margin}; font-size: {font_size}; padding: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{icon} {title}</h4>", layout=widgets.Layout(width='100%', margin='0', padding='0'))
    
    elif element_type == 'container':
        children, container_type = content, kwargs.get('container_type', 'vbox')
        layout_params = {k: kwargs.get(k, v) for k, v in {'width': '100%', 'max_width': '100%', 'padding': '8px', 'margin': '0px', 'justify_content': 'flex-start', 'align_items': 'stretch', 'overflow': 'hidden'}.items()}
        layout = widgets.Layout(**layout_params)
        return widgets.HBox(children, layout=layout) if container_type.lower() == 'hbox' else widgets.VBox(children, layout=layout)
    
    elif element_type == 'two_column':
        left_content, right_content = content
        left_width, right_width, vertical_align = kwargs.get('left_width', '48%'), kwargs.get('right_width', '48%'), kwargs.get('vertical_align', 'flex-start')
        left_wrapper = widgets.VBox([left_content], layout=widgets.Layout(width=left_width, margin='0', padding='4px', overflow='hidden'))
        right_wrapper = widgets.VBox([right_content], layout=widgets.Layout(width=right_width, margin='0', padding='4px', overflow='hidden'))
        return widgets.HBox([left_wrapper, right_wrapper], layout=widgets.Layout(width='100%', max_width='100%', justify_content='space-between', align_items=vertical_align, margin='0', padding='0', overflow='hidden'))
    
    else:
        raise ValueError(f"Unsupported element_type: {element_type}")

def get_layout(layout_name: str, **overrides) -> widgets.Layout:
    """
    Get layout with optional parameter overrides in one line.
    
    Args:
        layout_name: Name of layout from LAYOUTS dict
        **overrides: Layout parameters to override
        
    Returns:
        Layout object with applied overrides
    """
    base_layout = LAYOUTS.get(layout_name, widgets.Layout())
    layout_dict = {attr: getattr(base_layout, attr) for attr in dir(base_layout) if not attr.startswith('_') and hasattr(base_layout, attr) and getattr(base_layout, attr) is not None}
    layout_dict.update(overrides)
    return widgets.Layout(**layout_dict)

def get_responsive_config(widget_type: str, **kwargs) -> Dict[str, Any]:
    """
    Get responsive configuration for different widget types in one line.
    
    Args:
        widget_type: Type of widget ('button', 'dropdown', 'input')
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with layout and style configuration
    """
    configs = {
        'button': {'layout': widgets.Layout(width=kwargs.get('width', 'auto'), max_width=kwargs.get('max_width', '150px'), height='32px', margin='2px', overflow='hidden')},
        'dropdown': {'layout': widgets.Layout(width=kwargs.get('width', '100%'), max_width='100%', margin='3px 0', overflow='hidden'), 'style': {'description_width': kwargs.get('description_width', '80px')}},
        'input': {'layout': widgets.Layout(width=kwargs.get('width', '100%'), max_width='100%', margin='3px 0', overflow='hidden')}
    }
    return configs.get(widget_type, {'layout': widgets.Layout(overflow='hidden', max_width='100%')})

def apply_responsive_fixes(widget: widgets.Widget, recursive: bool = True) -> widgets.Widget:
    """
    Apply responsive fixes to widget with optional recursion in one line.
    
    Args:
        widget: Widget to fix
        recursive: Whether to apply fixes to children
        
    Returns:
        Widget with responsive fixes applied
    """
    if hasattr(widget, 'layout'):
        widget.layout.max_width = widget.layout.max_width or '100%'
        widget.layout.overflow = widget.layout.overflow or 'hidden'
        # Fix excessive margins
        if hasattr(widget.layout, 'margin') and widget.layout.margin and 'px' in str(widget.layout.margin):
            try:
                margin_val = int(str(widget.layout.margin).replace('px', ''))
                widget.layout.margin = '10px' if margin_val > 20 else widget.layout.margin
            except: pass
    
    # Apply to children if recursive
    if recursive and hasattr(widget, 'children'):
        [apply_responsive_fixes(child, recursive) for child in widget.children]
    
    return widget

# =============================================================================
# CONVENIENCE ONE-LINER FUNCTIONS
# =============================================================================

# One-liner convenience functions for common operations
create_divider = lambda margin='15px 0', color='#eee', height='1px': create_element('divider', margin=margin, color=color, height=height)
create_section_header = lambda title, icon='📋', color='#333', font_size='16px', margin='15px 0 8px 0': create_element('header', title, icon=icon, color=color, font_size=font_size, margin=margin)
create_responsive_container = lambda children, container_type='vbox', **kwargs: create_element('container', children, container_type=container_type, **kwargs)
create_responsive_two_column = lambda left_content, right_content, **kwargs: create_element('two_column', [left_content, right_content], **kwargs)
get_responsive_button_layout = lambda width='auto', max_width='150px': get_responsive_config('button', width=width, max_width=max_width)['layout']
get_responsive_dropdown_layout = lambda width='100%', description_width='80px': get_responsive_config('dropdown', width=width, description_width=description_width)