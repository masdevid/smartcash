"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Save reset buttons dengan flex layout dan compact styling
"""

import ipywidgets as widgets
from typing import Dict, Any, Literal, Optional

def create_save_reset_buttons(
    save_label: str = "Simpan", 
    reset_label: str = "Reset", 
    button_width: str = '100px',
    container_width: str = '100%', 
    save_tooltip: str = "Simpan konfigurasi saat ini",
    reset_tooltip: str = "Reset ke nilai default", 
    with_sync_info: bool = False,
    sync_message: str = "Konfigurasi akan otomatis disinkronkan.",
    show_icons: bool = False,
    alignment: Literal['left', 'center', 'right'] = 'right'
) -> Dict[str, Any]:
    """Create save dan reset buttons dengan flex layout.
    
    Args:
        save_label: Label text for save button
        reset_label: Label text for reset button
        button_width: Width of each button
        container_width: Width of the container
        save_tooltip: Tooltip text for save button
        reset_tooltip: Tooltip text for reset button
        with_sync_info: Whether to show sync info message
        sync_message: Text for sync info message
        show_icons: Whether to show icons on buttons (False for text only)
        alignment: Button alignment ('left', 'center', or 'right')
        
    Returns:
        Dictionary containing container and button widgets
    """
    
    # Save button
    save_button = widgets.Button(
        description=save_label,
        button_style='primary', 
        tooltip=save_tooltip,
        icon='save' if show_icons else '',  # Only show icon if show_icons is True
        layout=widgets.Layout(width=button_width, height='30px', margin='0 4px 0 0')
    )
    
    # Reset button  
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',
        tooltip=reset_tooltip,
        icon='undo' if show_icons else '',  # Only show icon if show_icons is True
        layout=widgets.Layout(width=button_width, height='30px', margin='0')
    )
    
    # Map alignment parameter to justify-content CSS value
    justify_content_map = {
        'left': 'flex-start',
        'center': 'center',
        'right': 'flex-end'
    }
    justify_content = justify_content_map.get(alignment, 'flex-end')  # Default to right alignment
    
    # Flex container untuk buttons
    button_container = widgets.HBox([save_button, reset_button], 
        layout=widgets.Layout(
            width='auto', 
            display='flex',
            flex_flow='row nowrap',
            justify_content=justify_content,  # Use the mapped alignment value
            align_items='center', 
            gap='6px',
            margin='0', 
            padding='0'
        ))
    
    components = [button_container]
    sync_info_widget = None
    
    if with_sync_info and sync_message:
        sync_info_widget = widgets.HTML(f"""
        <div style='margin-top: 3px; font-style: italic; color: #666; text-align: right; 
                    font-size: 10px; line-height: 1.2; max-width: 100%; overflow: hidden; 
                    text-overflow: ellipsis;'>
            ℹ️ {sync_message}
        </div>""",
        layout=widgets.Layout(width='100%', margin='0'))
        components.append(sync_info_widget)
    
    container = widgets.VBox(components, 
        layout=widgets.Layout(
            width=container_width, 
            max_width='100%', 
            margin='4px 0', 
            padding='0', 
            overflow='hidden',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        ))
    
    return {
        'container': container,
        'save_button': save_button, 
        'reset_button': reset_button,
        'sync_info': sync_info_widget
    }