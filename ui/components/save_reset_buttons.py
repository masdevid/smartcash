"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Fixed save reset buttons dengan one-liner style dan responsive layout tanpa icon
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_save_reset_buttons(save_label: str = "Simpan", reset_label: str = "Reset", button_width: str = '100px',
                             container_width: str = '100%', save_tooltip: str = "Simpan konfigurasi saat ini",
                             reset_tooltip: str = "Reset ke nilai default", with_sync_info: bool = False,
                             sync_message: str = "Konfigurasi akan otomatis disinkronkan dengan Google Drive.") -> Dict[str, Any]:
    """Create save dan reset buttons dengan one-liner responsive layout tanpa icon."""
    
    # Save button - secondary style, no icon
    save_button = widgets.Button(
        description=save_label, 
        button_style='primary', 
        tooltip=save_tooltip,
        layout=widgets.Layout(width=button_width, height='32px', margin='0 5px 0 0')
    )
    
    # Reset button - default style, no icon  
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',  # Default/grey style
        tooltip=reset_tooltip,
        layout=widgets.Layout(width=button_width, height='32px', margin='0')
    )
    
    # Container untuk buttons
    button_container = widgets.HBox([save_button, reset_button], 
        layout=widgets.Layout(width='auto', justify_content='flex-end', 
                             align_items='center', margin='0', padding='0'))
    
    components = [button_container]
    sync_info_widget = None
    
    if with_sync_info and sync_message:
        sync_info_widget = widgets.HTML(f"""
        <div style='margin-top: 4px; font-style: italic; color: #666; text-align: left; 
                    font-size: 11px; line-height: 1.3; max-width: 100%; overflow: hidden; 
                    text-overflow: ellipsis;'>
            ℹ️ {sync_message}
        </div>""",
        layout=widgets.Layout(width='100%', margin='0'))
        components.append(sync_info_widget)
    
    container = widgets.VBox(components, 
        layout=widgets.Layout(width=container_width, max_width='100%', margin='5px 0', 
                             padding='0', overflow='hidden'))
    
    return {
        'container': container,
        'save_button': save_button, 
        'reset_button': reset_button,
        'sync_info': sync_info_widget
    }