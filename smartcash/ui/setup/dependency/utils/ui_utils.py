"""
UI Utility Functions for Dependency Management

This module provides utility functions for creating and managing UI components
used in the dependency management interface.
"""
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML
from enum import Enum

class ButtonStyle(str, Enum):
    """Button style options"""
    PRIMARY = 'primary'
    SUCCESS = 'success'
    INFO = 'info'
    WARNING = 'warning'
    DANGER = 'danger'
    SECONDARY = ''

class StatusType(str, Enum):
    """Status type options"""
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'


def create_header(title: str, description: str = "", icon: str = "", **style) -> widgets.Widget:
    """Create a styled header component.
    
    Args:
        title: Header title text
        description: Optional description text
        icon: Optional icon to display before title
        **style: Additional style parameters
        
    Returns:
        widgets.VBox containing the header
    """
    styles = {
        'container': {
            'margin': '0 0 16px 0',
            'padding': '12px 16px',
            'border_radius': '8px',
            'background': '#f8f9fa',
            'border': '1px solid #e0e0e0',
        },
        'title': {
            'font_size': '20px',
            'font_weight': '600',
            'margin': '0 0 4px 0',
            'color': '#333',
        },
        'description': {
            'font_size': '14px',
            'color': '#666',
            'margin': '0',
        },
        **{k: v for k, v in style.items() if k != 'container'}
    }
    
    title_html = f"{icon} {title}" if icon else title
    title_widget = widgets.HTML(f"<div style='{_dict_to_css(styles['title'])}'>{title_html}</div>")
    
    children = [title_widget]
    if description:
        desc_widget = widgets.HTML(f"<p style='{_dict_to_css(styles['description'])}'>{description}</p>")
        children.append(desc_widget)
    
    return widgets.VBox(
        children=children,
        layout=widgets.Layout(**styles.get('container', {}))
    )


def create_status_panel(message: str, status_type: str = "info", **style) -> widgets.Widget:
    """Create a status panel with appropriate styling based on status type.
    
    Args:
        message: Status message to display
        status_type: One of 'info', 'success', 'warning', 'error'
        **style: Additional style parameters
        
    Returns:
        widgets.HTML containing the status panel
    """
    status_colors = {
        'info': {'bg': '#e7f5ff', 'border': '#74c0fc', 'text': '#1864ab'},
        'success': {'bg': '#ebfbee', 'border': '#40c057', 'text': '#2b8a3e'},
        'warning': {'bg': '#fff3bf', 'border': '#ffd43b', 'text': '#e67700'},
        'error': {'bg': '#ffe3e3', 'border': '#ff8787', 'text': '#c92a2a'},
    }
    
    colors = status_colors.get(status_type.lower(), status_colors['info'])
    
    styles = {
        'container': {
            'padding': '12px 16px',
            'border_radius': '8px',
            'border': f'1px solid {colors["border"]}',
            'background': colors['bg'],
            'margin': '0 0 16px 0',
            'color': colors['text'],
            'font_size': '14px',
            'display': 'flex',
            'align_items': 'center',
            'gap': '8px',
        },
        **style
    }
    
    icon_map = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
    }
    icon = icon_map.get(status_type.lower(), 'ℹ️')
    
    return widgets.HTML(
        f"""
        <div style='{css}'>
            <span style='font-size: 16px;'>{icon}</span>
            <span>{message}</span>
        </div>
        """.format(css=_dict_to_css(styles['container']))
    )


def create_card(title: str, content: widgets.Widget, **style) -> widgets.Widget:
    """Create a card container with title and content.
    
    Args:
        title: Card title
        content: Widget to display inside the card
        **style: Additional style parameters
        
    Returns:
        widgets.VBox containing the card
    """
    styles = {
        'container': {
            'border': '1px solid #e0e0e0',
            'border_radius': '8px',
            'overflow': 'hidden',
            'margin': '0 0 16px 0',
            'background': '#ffffff',
            'box_shadow': '0 1px 3px rgba(0,0,0,0.05)',
        },
        'header': {
            'padding': '12px 16px',
            'background': '#f8f9fa',
            'border_bottom': '1px solid #e0e0e0',
            'font_weight': '600',
            'color': '#333',
        },
        'body': {
            'padding': '16px',
        },
        **style
    }
    
    header = widgets.HTML(f"<div style='{_dict_to_css(styles['header'])}'>{title}</div>")
    
    return widgets.VBox(
        children=[header, content],
        layout=widgets.Layout(**styles['container'])
    )


def create_section(title: str, content: widgets.Widget, **style) -> widgets.Widget:
    """Create a section with title and content.
    
    Args:
        title: Section title
        content: Widget to display in the section
        **style: Additional style parameters
        
    Returns:
        widgets.VBox containing the section
    """
    styles = {
        'container': {
            'margin': '0 0 24px 0',
        },
        'title': {
            'font_size': '16px',
            'font_weight': '600',
            'margin': '0 0 8px 0',
            'color': '#333',
        },
        'content': {
            'margin': '0',
        },
        **style
    }
    
    title_widget = widgets.HTML(f"<div style='{_dict_to_css(styles['title'])}'>{title}</div>")
    
    return widgets.VBox(
        children=[
            title_widget,
            widgets.Box(
                [content],
                layout=widgets.Layout(**styles['content'])
            )
        ],
        layout=widgets.Layout(**styles['container'])
    )


def create_action_buttons(
    primary_button: Dict[str, Any],
    secondary_buttons: List[Dict[str, Any]],
    **style
) -> widgets.Widget:
    """Create a row of action buttons.
    
    Args:
        primary_button: Configuration for primary button
        secondary_buttons: List of configurations for secondary buttons
        **style: Additional style parameters
        
    Returns:
        widgets.HBox containing the buttons
    """
    styles = {
        'container': {
            'margin': '16px 0',
            'gap': '8px',
            'align_items': 'center',
        },
        'button': {
            'button_width': 'auto',
            'padding': '0 16px',
        },
        **style
    }
    
    # Create primary button
    primary_btn = widgets.Button(
        description=primary_button.get('label', ''),
        icon=primary_button.get('icon', ''),
        tooltip=primary_button.get('tooltip', ''),
        button_style=primary_button.get('style', 'primary'),
        disabled=primary_button.get('disabled', False),
        layout=widgets.Layout(
            width=styles['button']['button_width'],
            padding=styles['button']['padding'],
            height='34px',
        )
    )
    
    # Create secondary buttons
    secondary_btns = []
    for btn_config in secondary_buttons:
        btn = widgets.Button(
            description=btn_config.get('label', ''),
            icon=btn_config.get('icon', ''),
            tooltip=btn_config.get('tooltip', ''),
            button_style=btn_config.get('style', ''),
            disabled=btn_config.get('disabled', False),
            layout=widgets.Layout(
                width=styles['button']['button_width'],
                padding=styles['button']['padding'],
                height='34px',
            )
        )
        secondary_btns.append(btn)
    
    return widgets.HBox(
        children=[primary_btn, *secondary_btns],
        layout=widgets.Layout(**styles['container'])
    )


def _dict_to_css(style_dict: Dict[str, Any]) -> str:
    """Convert a dictionary of styles to a CSS string.
    
    Args:
        style_dict: Dictionary of CSS properties and values
        
    Returns:
        CSS string
    """
    return '; '.join([f"{k.replace('_', '-')}: {v}" for k, v in style_dict.items()])


def create_tabs(tab_contents: List[Tuple[str, widgets.Widget]], **style) -> widgets.Tab:
    """Create a tabbed interface with the given content.
    
    Args:
        tab_contents: List of (title, widget) tuples for each tab
        **style: Additional style parameters
        
    Returns:
        widgets.Tab containing the tabbed interface
    """
    styles = {
        'container': {
            'width': '100%',
            'margin': '0 0 16px 0',
        },
        'tab': {
            'padding': '12px',
        },
        **style
    }
    
    tabs = widgets.Tab(layout=widgets.Layout(**styles['container']))
    tabs.children = [content for _, content in tab_contents]
    for i, (title, _) in enumerate(tab_contents):
        tabs.set_title(i, title)
    
    return tabs


def create_log_accordion(
    module_name: str = "Logs",
    height: str = "200px",
    **style
) -> widgets.Accordion:
    """Create an accordion for displaying logs.
    
    Args:
        module_name: Name to display in the accordion header
        height: Height of the log output area
        **style: Additional style parameters
        
    Returns:
        widgets.Accordion containing the log output
    """
    styles = {
        'container': {
            'width': '100%',
            'margin': '16px 0',
        },
        'output': {
            'width': '100%',
            'height': height,
            'overflow_y': 'auto',
            'background': '#f8f9fa',
            'border': '1px solid #e0e0e0',
            'border_radius': '4px',
            'padding': '8px',
            'font_family': 'monospace',
            'font_size': '12px',
        },
        **style
    }
    
    output = widgets.Output(layout=widgets.Layout(**styles['output']))
    accordion = widgets.Accordion(children=[output], layout=widgets.Layout(**styles['container']))
    accordion.set_title(0, f"📋 {module_name}")
    accordion.selected_index = None  # Start collapsed
    
    return accordion


def create_save_reset_buttons(
    save_label: str = "💾 Simpan",
    reset_label: str = "🔄 Reset",
    **style
) -> widgets.HBox:
    """Create save and reset buttons with consistent styling.
    
    Args:
        save_label: Text for the save button
        reset_label: Text for the reset button
        **style: Additional style parameters
        
    Returns:
        widgets.HBox containing the buttons
    """
    styles = {
        'container': {
            'display': 'flex',
            'justify_content': 'flex-end',
            'gap': '8px',
            'margin': '16px 0',
        },
        'button': {
            'width': '120px',
        },
        **style
    }
    
    save_btn = widgets.Button(
        description=save_label,
        button_style=ButtonStyle.SUCCESS,
        layout=widgets.Layout(**styles['button'])
    )
    
    reset_btn = widgets.Button(
        description=reset_label,
        button_style=ButtonStyle.SECONDARY,
        layout=widgets.Layout(**styles['button'])
    )
    
    return widgets.HBox(
        [reset_btn, save_btn],
        layout=widgets.Layout(**styles['container'])
    )


def create_checkbox(
    description: str = "",
    value: bool = False,
    indent: bool = True,
    **style
) -> widgets.Checkbox:
    """Create a styled checkbox with consistent look and feel.
    
    Args:
        description: Label text for the checkbox
        value: Initial value of the checkbox
        indent: Whether to add left margin to the description
        **style: Additional style parameters
        
    Returns:
        widgets.Checkbox with applied styles
    """
    styles = {
        'container': {
            'margin': '4px 0',
        },
        'description_width': 'initial',
        'indent': '20px' if indent else '0',
        'padding': '6px 0',
        **style
    }
    
    return widgets.Checkbox(
        value=value,
        description=description,
        indent=False,
        layout=widgets.Layout(
            margin=styles['container']['margin'],
            padding=styles['padding']
        ),
        style={
            'description_width': styles['description_width']
        },
        description_tooltip=style.get('tooltip', '')
    )


def create_text_input(
    placeholder: str = "",
    value: str = "",
    **style
) -> widgets.Text:
    """Create a styled text input field.
    
    Args:
        placeholder: Placeholder text
        value: Initial value
        **style: Additional style parameters
        
    Returns:
        widgets.Text with applied styles
    """
    styles = {
        'container': {
            'width': '100%',
            'margin': '4px 0 12px 0',
        },
        'input': {
            'padding': '8px 12px',
            'border_radius': '4px',
            'border': '1px solid #ced4da',
        },
        **style
    }
    
    return widgets.Text(
        value=value,
        placeholder=placeholder,
        layout=widgets.Layout(
            width='100%',
            margin=styles['container']['margin'],
        ),
        style={
            'description_width': 'initial',
        },
        **styles.get('input', {})
    )


def create_package_checkbox(
    package_name: str,
    version: str = "",
    is_installed: bool = False,
    on_change: Optional[Callable[[str, bool], None]] = None,
    **style
) -> widgets.HBox:
    """Create a styled package checkbox with version and status indicator.
    
    Args:
        package_name: Name of the package
        version: Package version (optional)
        is_installed: Whether the package is installed
        on_change: Callback when checkbox state changes
        **style: Additional style parameters
        
    Returns:
        widgets.HBox containing the package checkbox with version and status
    """
    # Create the main checkbox
    checkbox = widgets.Checkbox(
        value=is_installed,
        indent=False,
        layout=widgets.Layout(width='20px', margin='0 8px 0 0')
    )
    
    # Create version label
    version_text = f" ({version})" if version else ""
    version_label = widgets.HTML(
        f"<span style='color: #666; font-size: 0.9em;'>{version_text}</span>",
        layout=widgets.Layout(margin='0 8px 0 0')
    )
    
    # Create status indicator
    status_emoji = "✅" if is_installed else "❌"
    status_tooltip = "Terinstall" if is_installed else "Belum terinstall"
    status = widgets.HTML(
        f"<span title='{status_tooltip}'>{status_emoji}</span>",
        layout=widgets.Layout(margin='0 8px 0 0', width='20px')
    )
    
    # Package name label
    name_label = widgets.HTML(
        f"<span style='font-family: monospace;'>{package_name}</span>"
    )
    
    # Create the container
    container = widgets.HBox(
        [checkbox, name_label, version_label, status],
        layout=widgets.Layout(
            margin='4px 0',
            padding='4px 8px',
            border_radius='4px',
            border='1px solid #e0e0e0',
            width='100%',
            **style.get('container', {})
        )
    )
    
    # Add hover effect
    container.add_class('package-checkbox-container')
    
    # Handle checkbox changes
    def on_checkbox_change(change):
        if on_change:
            on_change(package_name, change['new'])
    
    checkbox.observe(on_checkbox_change, names='value')
    
    return container


def create_dropdown(
    options: List[Union[str, Tuple[str, str]]],
    value: Optional[str] = None,
    description: str = "",
    **style
) -> widgets.Dropdown:
    """Create a styled dropdown menu.
    
    Args:
        options: List of options or (value, label) tuples
        value: Initial selected value
        description: Label text
        **style: Additional style parameters
        
    Returns:
        widgets.Dropdown with applied styles
    """
    styles = {
        'container': {
            'width': '100%',
            'margin': '4px 0 12px 0',
        },
        'description_width': 'initial',
        **style
    }
    
    return widgets.Dropdown(
        options=options,
        value=value if value else (options[0] if isinstance(options[0], str) else options[0][0]),
        description=description,
        disabled=False,
        layout=widgets.Layout(
            width='100%',
            margin=styles['container']['margin']
        ),
        style={
            'description_width': styles['description_width']
        },
        **{k: v for k, v in styles.items() if k not in ['container', 'description_width']}
    )
