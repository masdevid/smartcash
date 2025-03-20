"""
File: smartcash/ui/utils/info_utils.py
Deskripsi: Utilitas untuk membuat dan mengelola info box dengan accordion
"""

import ipywidgets as widgets
from typing import Optional, Union

from smartcash.ui.utils.constants import ALERT_STYLES

def create_info_accordion(
    title: str, 
    content: Union[str, widgets.Widget], 
    style: str = "info", 
    icon: Optional[str] = None,
    open_by_default: bool = False
) -> widgets.Accordion:
    """
    Buat accordion dengan info box yang dapat di-collapse.
    
    Args:
        title: Judul accordion
        content: Konten dalam bentuk HTML string atau widget
        style: Style alert ('info', 'success', 'warning', 'error')
        icon: Icon opsional (emoji)
        open_by_default: Buka accordion secara default
        
    Returns:
        Widget accordion yang berisi info box
    """
    # Dapatkan style dari constants
    style_config = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    
    # Set icon
    if icon is None:
        icon = style_config['icon']
    
    # Siapkan judul dengan icon
    title_with_icon = f"{icon} {title}"
    
    # Siapkan konten
    if isinstance(content, str):
        # Jika konten berupa string, bungkus dalam HTML widget
        styled_content = f"""
        <div style="padding: 10px; 
                    background-color: {style_config['bg_color']}; 
                    color: {style_config['text_color']}; 
                    border-radius: 5px;">
            {content}
        </div>
        """
        content_widget = widgets.HTML(value=styled_content)
    else:
        # Jika konten berupa widget, gunakan langsung
        content_widget = content
    
    # Buat accordion
    accordion = widgets.Accordion([content_widget])
    accordion.set_title(0, title_with_icon)
    
    # Set apakah terbuka secara default
    if open_by_default:
        accordion.selected_index = 0
    else:
        accordion.selected_index = None
    
    return accordion

def style_info_content(
    content: str, 
    style: str = "info", 
    padding: int = 10, 
    border_radius: int = 5
) -> str:
    """
    Styling konten HTML untuk info box.
    
    Args:
        content: Konten HTML
        style: Style alert ('info', 'success', 'warning', 'error')
        padding: Padding dalam pixel
        border_radius: Border radius dalam pixel
        
    Returns:
        Konten HTML yang telah di-styling
    """
    # Dapatkan style dari constants
    style_config = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    
    # Styling konten
    styled_content = f"""
    <div style="padding: {padding}px; 
                background-color: {style_config['bg_color']}; 
                color: {style_config['text_color']}; 
                border-radius: {border_radius}px;">
        {content}
    </div>
    """
    
    return styled_content

def create_tabbed_info(
    tabs_content: Dict[str, str], 
    style: str = "info"
) -> widgets.Tab:
    """
    Buat info box dengan multiple tabs.
    
    Args:
        tabs_content: Dictionary {tab_title: content}
        style: Style alert ('info', 'success', 'warning', 'error')
        
    Returns:
        Widget tab berisi multiple info box
    """
    # Dapatkan style dari constants
    style_config = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    
    # Buat widget untuk setiap tab
    tab_widgets = []
    tab_titles = []
    
    for title, content in tabs_content.items():
        styled_content = f"""
        <div style="padding: 10px; 
                    background-color: {style_config['bg_color']}; 
                    color: {style_config['text_color']}; 
                    border-radius: 5px;">
            {content}
        </div>
        """
        tab_widgets.append(widgets.HTML(value=styled_content))
        tab_titles.append(title)
    
    # Buat tabs
    tabs = widgets.Tab(children=tab_widgets)
    
    # Set judul tab
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
    
    return tabs