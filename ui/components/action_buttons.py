"""
File: smartcash/ui/components/action_buttons.py
Deskripsi: Action buttons dengan flex layout dan compact styling
"""

from typing import Dict, Any
import ipywidgets as widgets

def create_action_buttons(primary_label: str = "Download Dataset", primary_icon: str = "download", 
                         secondary_buttons: list = None, cleanup_enabled: bool = True,
                         cleanup_label: str = "Cleanup Dataset", cleanup_tooltip: str = "Hapus dataset yang sudah ada",
                         button_width: str = '140px', container_width: str = '100%',
                         primary_style: str = '') -> dict:
    """Create action buttons dengan flex layout"""
    secondary_buttons = secondary_buttons or [("Check Dataset", "search", "info")]
    
    # Primary button
    download_button = widgets.Button(
        description=primary_label, 
        button_style=primary_style or 'primary', 
        tooltip=f'Klik untuk {primary_label.lower()}',
        icon=primary_icon if primary_icon in ['download', 'save', 'search', 'play'] else '',
        layout=widgets.Layout(width=button_width, height='32px', margin='0')
    )
    setattr(download_button, '_original_style', primary_style or 'primary')
    setattr(download_button, '_original_description', primary_label)
    
    # Secondary button
    if secondary_buttons and len(secondary_buttons) > 0:
        secondary_label, secondary_icon, secondary_style = secondary_buttons[0]
    else:
        secondary_label, secondary_icon, secondary_style = ("Check", "search", "info")
    
    check_button = widgets.Button(
        description=secondary_label,
        button_style=secondary_style,
        tooltip=f'Klik untuk {secondary_label.lower()}',
        icon=secondary_icon if secondary_icon in ['search', 'info', 'check', 'clipboard'] else '',
        layout=widgets.Layout(width=button_width, height='32px', margin='0')
    )
    setattr(check_button, '_original_style', secondary_style)
    setattr(check_button, '_original_description', secondary_label)
    
    # Cleanup button
    cleanup_button = None
    if cleanup_enabled:
        cleanup_button = widgets.Button(
            description=cleanup_label,
            button_style='warning',
            tooltip=cleanup_tooltip,
            icon='trash',
            layout=widgets.Layout(width=button_width, height='32px', margin='0')
        )
        setattr(cleanup_button, '_original_style', 'warning')
        setattr(cleanup_button, '_original_description', cleanup_label)
    
    # Button list
    button_list = [download_button, check_button]
    if cleanup_button:
        button_list.append(cleanup_button)
    
    # Flex container
    container = widgets.HBox(
        button_list,
        layout=widgets.Layout(
            width=container_width,
            display='flex',
            flex_flow='row wrap',
            justify_content='flex-start',
            align_items='center',
            gap='8px',
            margin='8px 0'
        )
    )
    
    result = {
        'container': container,
        'download_button': download_button,
        'check_button': check_button,
        'buttons': button_list
    }
    
    if cleanup_button:
        result['cleanup_button'] = cleanup_button
    
    return result