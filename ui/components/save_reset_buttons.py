"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Save reset buttons dengan flex layout dan compact styling
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_save_reset_buttons(save_label: str = "Simpan", reset_label: str = "Reset", button_width: str = '100px',
                             container_width: str = '100%', save_tooltip: str = "Simpan konfigurasi saat ini",
                             reset_tooltip: str = "Reset ke nilai default", with_sync_info: bool = False,
                             sync_message: str = "Konfigurasi akan otomatis disinkronkan.") -> Dict[str, Any]:
    """Create save dan reset buttons dengan flex layout."""
    
    # Save button
    save_button = widgets.Button(
        description=save_label, 
        button_style='primary', 
        tooltip=save_tooltip,
        layout=widgets.Layout(width=button_width, height='30px', margin='0 4px 0 0')
    )
    
    # Reset button  
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',
        tooltip=reset_tooltip,
        layout=widgets.Layout(width=button_width, height='30px', margin='0')
    )
    
    # Flex container untuk buttons
    button_container = widgets.HBox([save_button, reset_button], 
        layout=widgets.Layout(
            width='auto', 
            display='flex',
            flex_flow='row nowrap',
            justify_content='flex-end', 
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