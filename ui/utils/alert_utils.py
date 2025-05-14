"""
File: smartcash/ui/utils/alert_utils.py
Deskripsi: Komponen UI untuk alerts, info boxes, dan status indicators
"""

import ipywidgets as widgets
from IPython.display import HTML
from typing import Optional, Union

from smartcash.ui.utils.constants import ALERT_STYLES, COLORS

def create_status_indicator(status: str, message: str) -> HTML:
    """Create a styled status indicator.

    Args:
        status: Status type ('info', 'success', 'warning', 'error')
        message: Status message

    Returns:
        HTML widget with styled status indicator
    """
    style = ALERT_STYLES.get(status, ALERT_STYLES['info'])
    html_content = (
        f'<div style="margin: 5px 0; padding: 8px 12px; '
        f'border-radius: 4px; background-color: {COLORS["light"]};">'
        f'<span style="color: {style["text_color"]}; font-weight: bold;">'
        f'{style["icon"]} {message}'
        f'</span></div>'
    )
    return HTML(html_content)

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