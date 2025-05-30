"""
File: smartcash/ui/utils/alert_utils.py
Deskripsi: Komponen UI untuk alerts, info boxes, dan status indicators
"""

import ipywidgets as widgets
from IPython.display import HTML
from typing import Optional, Union

from smartcash.ui.utils.constants import ALERT_STYLES, COLORS

def create_status_indicator(status: str, message: str, icon: Optional[str] = None) -> HTML:
    """Create a styled status indicator.

    Args:
        status: Status type ('info', 'success', 'warning', 'error')
        message: Status message
        icon: Optional custom emoji icon (defaults to type-specific icon)

    Returns:
        HTML widget with styled status indicator
    """
    style = ALERT_STYLES.get(status, ALERT_STYLES['info'])
    icon_str = icon or style["icon"]
    
    html_content = (
        f'<div style="margin: 5px 0; padding: 8px 12px; '
        f'border-radius: 4px; background-color: {COLORS["light"]};">'
        f'<span style="color: {style["text_color"]}; font-weight: bold;">'
        f'{icon_str} {message}'
        f'</span></div>'
    )
    return HTML(html_content)

def create_error_alert(message: str, title: str = "Error") -> widgets.HTML:
    """Create a styled error alert.

    Args:
        message: Error message
        title: Alert title

    Returns:
        HTML widget with styled error alert
    """
    style = ALERT_STYLES['error']
    
    html_content = (
        f'<div style="padding: 10px; background-color: {style["bg_color"]}; '
        f'color: {style["text_color"]}; border-left: 4px solid {style["border_color"]}; '
        f'border-radius: 5px; margin: 10px 0;">'
        f'<h4 style="margin-top: 0; color: inherit;">{style["icon"]} {title}</h4>'
        f'<p>{message}</p>'
        f'</div>'
    )
    return widgets.HTML(value=html_content)

def create_info_alert(message: str, alert_type: str = 'info', icon: Optional[str] = None) -> widgets.HTML:
    """Create a styled alert box.

    Args:
        message: Alert message
        alert_type: Alert type ('info', 'success', 'warning', 'error')
        icon: Optional custom emoji icon (defaults to type-specific icon)

    Returns:
        HTML widget with styled alert
    """
    style = ALERT_STYLES.get(alert_type, ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    
    html_content = (
        f'<div style="padding: 10px; background-color: {style["bg_color"]}; '
        f'color: {style["text_color"]}; border-left: 4px solid {style["border_color"]}; '
        f'border-radius: 5px; margin: 4px 0;">'
        f'<div style="display: flex; align-items: flex-start;">'
        f'<div style="margin-right: 10px; font-size: 1.2em;">{icon_str}</div>'
        f'<div>{message}</div>'
        f'</div></div>'
    )
    return widgets.HTML(value=html_content)

def create_info_log(message: str, alert_type: str = 'info', icon: Optional[str] = None) -> widgets.HTML:
    """Create a styled plain log.

    Args:
        message: Log message
        alert_type: Log type ('info', 'success', 'warning', 'error')
        icon: Optional custom emoji icon (defaults to type-specific icon)

    Returns:
        HTML widget with styled log
    """
    style = ALERT_STYLES.get(alert_type, ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    
    html_content = (
        f'<div style="padding: 5px 10px; background-color: {style["bg_color"]}; '
        f'color: {style["text_color"]};'
        f'border-radius: 5px; margin: 1px 0;">'
        f'<div style="display: flex; align-items: flex-start;">'
        f'<div style="margin-right: 10px; font-size: 1.2em;">{icon_str}</div>'
        f'<div>{message}</div>'
        f'</div></div>'
    )
    return widgets.HTML(value=html_content)

def create_info_box(title: str, content: str, style: str = 'info', 
                   icon: Optional[str] = None, collapsed: bool = False) -> Union[widgets.HTML, widgets.Accordion]:
    """Create a collapsible info box.

    Args:
        title: Info box title
        content: HTML content for the info box
        style: Style type ('info', 'success', 'warning', 'error')
        icon: Optional custom emoji icon
        collapsed: Whether the box is collapsed by default

    Returns:
        HTML widget or Accordion with styled info box
    """
    style = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    title_with_icon = f"{icon_str} {title}"

    if collapsed:
        accordion = widgets.Accordion([widgets.HTML(value=content)], selected_index=None)
        accordion.set_title(0, title_with_icon)
        return accordion

    html_content = (
        f'<div style="padding: 10px; background-color: {style["bg_color"]}; '
        f'border-left: 4px solid {style["border_color"]}; color: {style["text_color"]}; '
        f'margin: 10px 0; border-radius: 4px;">'
        f'<h4 style="margin-top: 0; color: inherit;">{title_with_icon}</h4>'
        f'{content}'
        f'</div>'
    )
    return widgets.HTML(value=html_content)

def create_alert_html(message: str, alert_type: str = 'info', icon: Optional[str] = None) -> str:
    """Create a styled alert HTML string.

    Args:
        message: Alert message
        alert_type: Alert type ('info', 'success', 'warning', 'error')
        icon: Optional custom emoji icon (defaults to type-specific icon)

    Returns:
        HTML string with styled alert
    """
    style = ALERT_STYLES.get(alert_type, ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    
    html_content = (
        f'<div style="padding: 10px; background-color: {style["bg_color"]}; '
        f'color: {style["text_color"]}; border-left: 4px solid {style["border_color"]}; '
        f'border-radius: 5px; margin: 4px 0;">'
        f'<div style="display: flex; align-items: flex-start;">'
        f'<div style="margin-right: 10px; font-size: 1.2em;">{icon_str}</div>'
        f'<div>{message}</div>'
        f'</div></div>'
    )
    return html_content

def update_status_panel(panel: widgets.HTML, message: str, status_type: str = "info") -> None:
    """
    Update status panel yang sudah ada.
    
    Args:
        panel: Widget HTML status panel
        message: Pesan baru
        status_type: Tipe status baru ('info', 'success', 'warning', 'error')
    """
    # Dapatkan style berdasarkan tipe status
    style_info = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    bg_color = style_info['bg_color']
    text_color = style_info['text_color'] 
    icon = style_info['icon']
    
    # Update HTML
    panel.value = f"""
    <div style="padding:10px; background-color:{bg_color}; 
               color:{text_color}; border-radius:4px; margin:5px 0;
               border-left:4px solid {text_color};">
        <p style="margin:5px 0">{icon} {message}</p>
    </div>
    """