"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Fixed save reset buttons dengan responsive layout dan sync info
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS

def create_save_reset_buttons(
    save_label: str = "Simpan",
    reset_label: str = "Reset", 
    button_width: str = '90px',
    container_width: str = '100%',
    save_tooltip: str = "Simpan konfigurasi saat ini",
    reset_tooltip: str = "Reset ke nilai default",
    with_sync_info: bool = False,
    sync_message: str = "Konfigurasi akan otomatis disinkronkan dengan Google Drive."
) -> Dict[str, Any]:
    """
    Create save dan reset buttons dengan responsive layout dan optional sync info.
    
    Args:
        save_label: Label untuk save button
        reset_label: Label untuk reset button
        button_width: Lebar setiap button
        container_width: Lebar container
        save_tooltip: Tooltip untuk save button
        reset_tooltip: Tooltip untuk reset button
        with_sync_info: Apakah menampilkan sync info
        sync_message: Pesan sync info
        
    Returns:
        Dictionary berisi save dan reset button components dengan responsive layout
    """
    
    # Safe icon access
    def get_safe_icon(key: str, fallback: str) -> str:
        try:
            return ICONS.get(key, fallback)
        except (NameError, AttributeError):
            return fallback
    
    # 💾 Save button dengan responsive layout
    save_button = widgets.Button(
        description=save_label,
        button_style='primary',
        tooltip=save_tooltip,
        icon='save',
        layout=widgets.Layout(
            width=button_width, 
            height='30px',
            margin='0 5px 0 0'
        )
    )
    
    # 🔄 Reset button dengan responsive layout
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',
        tooltip=reset_tooltip,
        icon='refresh',
        layout=widgets.Layout(
            width=button_width, 
            height='30px',
            margin='0'
        )
    )
    
    # Button container dengan responsive layout
    button_container = widgets.HBox(
        [save_button, reset_button],
        layout=widgets.Layout(
            width='auto',
            justify_content='flex-end',
            align_items='center',
            margin='0',
            padding='0'
        )
    )
    
    # Components untuk return
    components = [button_container]
    
    # Sync info jika diminta
    sync_info_widget = None
    if with_sync_info and sync_message:
        sync_info_widget = widgets.HTML(
            value=f"""
            <div style='margin-top: 4px; font-style: italic; color: #666; 
                        text-align: right; font-size: 11px; line-height: 1.3;
                        max-width: 100%; overflow: hidden; text-overflow: ellipsis;'>
                {get_safe_icon('info', 'ℹ️')} {sync_message}
            </div>
            """,
            layout=widgets.Layout(width='100%', margin='0')
        )
        components.append(sync_info_widget)
    
    # Main container dengan responsive layout (NO HORIZONTAL SCROLL)
    container = widgets.VBox(
        components,
        layout=widgets.Layout(
            width=container_width,
            max_width='100%',
            margin='5px 0',
            padding='0',
            overflow='hidden'  # Prevent horizontal scroll
        )
    )
    
    return {
        'container': container,
        'save_button': save_button,      # Key konsisten dengan handlers
        'reset_button': reset_button,    # Key konsisten dengan handlers
        'sync_info': sync_info_widget    # Optional sync info widget
    }