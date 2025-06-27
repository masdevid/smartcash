"""
Reusable UI components for dependency management.

This module provides optimized, one-liner components for common UI patterns.
"""
from typing import Dict, Any, List, Optional, Callable, Union
import ipywidgets as widgets
from IPython.display import HTML, display

def create_card(content: Union[widgets.Widget, str], 
               title: str = "", 
               icon: str = "", 
               **style) -> widgets.VBox:
    """Create a card component with optional title and icon."""
    header = (f"<div style='font-weight:600;margin-bottom:8px'>{icon} {title}</div>" 
              if title or icon else "")
    content = widgets.HTML(content) if isinstance(content, str) else content
    return widgets.VBox(
        [widgets.HTML(header), content],
        layout=widgets.Layout(
            padding='12px',
            margin='0 0 16px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            **style
        )
    )

def create_button(text: str, 
                 style: str = 'primary', 
                 on_click: Optional[Callable] = None, 
                 **kwargs) -> widgets.Button:
    """Create a styled button with one-liner syntax."""
    styles = {
        'primary': {'button_color': '#007bff', 'text_color': 'white'},
        'secondary': {'button_color': '#6c757d', 'text_color': 'white'},
        'success': {'button_color': '#28a745', 'text_color': 'white'},
        'danger': {'button_color': '#dc3545', 'text_color': 'white'},
    }.get(style, {'button_color': '#f8f9fa'})
    
    btn = widgets.Button(
        description=text,
        button_style=style,
        layout=widgets.Layout(
            margin='2px',
            padding='6px 12px',
            **{k: v for k, v in styles.items() if k != 'button_style'}
        ),
        **kwargs
    )
    if on_click:
        btn.on_click(lambda b: on_click())
    return btn

def create_status(text: str, 
                 status: str = 'info', 
                 icon: bool = True) -> widgets.HTML:
    """Create a status indicator with one-liner syntax."""
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'loading': '⏳'
    }
    icon_str = f"{icons.get(status, '')} " if icon else ""
    return widgets.HTML(
        f"<div style='padding:8px;margin:4px 0;border-radius:4px;background:#f8f9fa;'>"
        f"{icon_str}{text}</div>"
    )

def create_package_card(pkg: Dict[str, Any], 
                       selected: bool = False,
                       on_toggle: Optional[Callable[[str, bool], None]] = None) -> widgets.HBox:
    """Create a one-liner package card component."""
    pkg_key = pkg.get('key', '')
    name = pkg.get('name', 'Unknown')
    version = f"v{pkg.get('version', '')}" if pkg.get('version') else ''
    
    checkbox = widgets.Checkbox(
        value=selected,
        indent=False,
        layout=widgets.Layout(width='20px', margin='0 8px 0 0')
    )
    
    if on_toggle:
        checkbox.observe(
            lambda change: on_toggle(pkg_key, change['new']), 
            names='value'
        )
    
    return widgets.HBox([
        checkbox,
        widgets.HTML(
            f"<div style='line-height:1.4;'>"
            f"<div><b>{name}</b> <span style='color:#6c757d;font-size:0.9em'>{version}</span></div>"
            f"<div style='font-size:0.9em;color:#6c757d'>{pkg.get('description', '')}</div>"
            f"</div>"
        )
    ], layout=widgets.Layout(
        padding='8px',
        margin='2px 0',
        border='1px solid #e9ecef',
        border_radius='4px',
        width='100%',
        background='white'
    ))
