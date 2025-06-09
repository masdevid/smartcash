"""
File: smartcash/ui/dataset/augmentation/utils/style_utils.py
Deskripsi: Fungsi styling terkonsolidasi untuk augmentation components
"""

import ipywidgets as widgets
from typing import Dict, Any, Tuple, List

# Style constants dengan color mapping
COMPONENT_THEMES = {
    'basic': {'color': '#4caf50', 'bg': '#4caf5015', 'border': '#4caf5040'},
    'advanced': {'color': '#9c27b0', 'bg': '#9c27b015', 'border': '#9c27b040'}, 
    'types': {'color': '#2196f3', 'bg': '#2196f315', 'border': '#2196f340'},
    'normalization': {'color': '#f57c00', 'bg': '#ff980315', 'border': '#ff980340'},
    'info': {'color': '#17a2b8', 'bg': '#17a2b815', 'border': '#17a2b840'}
}

def create_styled_container(content_widget: widgets.Widget, title: str, theme: str = 'basic', 
                          width: str = '48%', height: str = None) -> widgets.VBox:
    """Create styled container dengan theme support"""
    theme_config = COMPONENT_THEMES.get(theme, COMPONENT_THEMES['basic'])
    
    header_html = f"""
    <div style="padding: 8px 12px; margin-bottom: 8px;
                background: linear-gradient(145deg, {theme_config['bg']} 0%, rgba(255,255,255,0.9) 100%);
                border-radius: 8px; border-left: 4px solid {theme_config['color']};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h5 style="color: #333; margin: 0; font-size: 14px; font-weight: 600;">
            {title}
        </h5>
    </div>
    """
    
    layout_props = {
        'width': width,
        'margin': '5px',
        'padding': '10px',
        'border': f'1px solid {theme_config["border"]}',
        'border_radius': '8px',
        'background_color': 'rgba(255,255,255,0.8)',
        'display': 'flex',
        'flex_flow': 'column',
        'align_items': 'stretch',
        'flex': f'1 1 {width.replace("%", "")}%' if '%' in width else '1 1 auto'
    }
    
    if height:
        layout_props['height'] = height
        
    return widgets.VBox([
        widgets.HTML(header_html),
        content_widget
    ], layout=widgets.Layout(**layout_props))

def create_info_panel(content: str, theme: str = 'info', width: str = '100%') -> widgets.HTML:
    """Create compact info panel dengan theme"""
    theme_config = COMPONENT_THEMES.get(theme, COMPONENT_THEMES['info'])
    
    return widgets.HTML(
        f"""
        <div style="padding: 6px 8px; background-color: {theme_config['bg']}; 
                    border-radius: 4px; margin: 6px 0; font-size: 10px;
                    border: 1px solid {theme_config['border']}; line-height: 1.3;">
            {content}
        </div>
        """,
        layout=widgets.Layout(width=width, margin='3px 0')
    )

def create_section_header(title: str, theme: str = 'basic', margin: str = '6px 0') -> widgets.HTML:
    """Create section header dengan theme color"""
    color = COMPONENT_THEMES.get(theme, COMPONENT_THEMES['basic'])['color']
    return widgets.HTML(f"<h6 style='color: {color}; margin: {margin};'>{title}</h6>")

def create_widget_grid(widgets_list: List[widgets.Widget], columns: int = 2, 
                      gap: str = '15px', align: str = 'stretch') -> widgets.HBox:
    """Create responsive widget grid"""
    return widgets.HBox(widgets_list, layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row nowrap',
        justify_content='space-between',
        align_items=align,
        gap=gap,
        margin='8px 0'
    ))

def style_form_widget(widget: widgets.Widget, description_width: str = '120px', 
                     widget_width: str = '95%') -> widgets.Widget:
    """Apply consistent styling to form widgets"""
    # Set style untuk widgets yang support description_width
    if hasattr(widget, 'style') and hasattr(widget, 'description'):
        try:
            widget.style = {'description_width': description_width}
        except Exception:
            pass  # Some widgets don't support style modification
    
    # Set layout width
    if hasattr(widget, 'layout'):
        widget.layout = widgets.Layout(width=widget_width)
    
    return widget

def create_tabbed_container(tabs_config: List[Tuple[str, widgets.Widget]], 
                          theme: str = 'basic') -> widgets.Tab:
    """Create styled tabbed container"""
    tab_widgets = []
    tab_titles = []
    
    for title, content in tabs_config:
        styled_content = widgets.VBox([
            create_section_header(title, theme),
            content
        ], layout=widgets.Layout(padding='5px'))
        
        tab_widgets.append(styled_content)
        tab_titles.append(title.split(' ')[-1])  # Extract emoji + last word
    
    tabs = widgets.Tab(children=tab_widgets)
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
    
    return tabs

def create_info_content(items: List[Tuple[str, str]], theme: str = 'info') -> str:
    """Create formatted info content dengan theme color"""
    color = COMPONENT_THEMES[theme]['color']
    
    content_lines = []
    for label, value in items:
        content_lines.append(f"• <strong style='color: {color};'>{label}:</strong> {value}")
    
    return f"""
    <strong style="color: {color}; margin-bottom:4px">ℹ️ {items[0][0].split(':')[0]} Info:</strong><br>
    {"<br>".join(content_lines)}
    """

def apply_flex_layout(container: widgets.Widget, direction: str = 'column',
                     justify: str = 'flex-start', align: str = 'stretch',
                     gap: str = '4px', padding: str = '10px') -> widgets.Widget:
    """Apply flex layout to container"""
    container.layout = widgets.Layout(
        padding=padding,
        width='100%',
        display='flex',
        flex_flow=f'{direction} nowrap',
        justify_content=justify,
        align_items=align,
        gap=gap
    )
    return container

# One-liner utilities
styled_container = lambda content, title, theme='basic', width='48%': create_styled_container(content, title, theme, width)
info_panel = lambda content, theme='info': create_info_panel(content, theme)
section_header = lambda title, theme='basic': create_section_header(title, theme)
widget_grid = lambda widgets_list, cols=2: create_widget_grid(widgets_list, cols)
style_widget = lambda widget, desc_width='120px': style_form_widget(widget, desc_width)
tabbed_container = lambda tabs_config, theme='basic': create_tabbed_container(tabs_config, theme)
flex_layout = lambda container, direction='column': apply_flex_layout(container, direction)