"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Fixed save reset buttons dengan sync info support dan fallback handling
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_save_reset_buttons(
    save_label: str = "Simpan",
    reset_label: str = "Reset", 
    button_width: str = '100px',
    container_width: str = '100%',
    save_tooltip: str = None,
    reset_tooltip: str = None,
    with_sync_info: bool = False,
    sync_message: str = None
) -> Dict[str, Any]:
    """
    Create save dan reset buttons dengan sync info support.
    
    Args:
        save_label: Label untuk save button
        reset_label: Label untuk reset button
        button_width: Lebar setiap button
        container_width: Lebar container
        save_tooltip: Tooltip untuk save button
        reset_tooltip: Tooltip untuk reset button
        with_sync_info: Apakah menampilkan sync info
        sync_message: Custom sync message
        
    Returns:
        Dictionary berisi save dan reset button components
    """
    
    # 💾 Save button
    save_button = widgets.Button(
        description=save_label,
        button_style='primary',
        tooltip=save_tooltip or 'Simpan konfigurasi saat ini',
        icon='save',
        layout=widgets.Layout(width=button_width, height='32px')
    )
    
    # 🔄 Reset button
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',
        tooltip=reset_tooltip or 'Reset ke nilai default',
        icon='refresh',
        layout=widgets.Layout(width=button_width, height='32px')
    )
    
    # Container elements
    container_elements = [save_button, reset_button]
    
    # Add sync info jika diminta
    sync_info_widget = None
    if with_sync_info:
        default_sync_msg = "Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset."
        sync_message = sync_message or default_sync_msg
        
        sync_info_widget = widgets.HTML(
            value=f"<div style='margin-top: 5px; font-style: italic; color: #666; text-align: right; font-size: 12px;'>ℹ️ {sync_message}</div>",
            layout=widgets.Layout(width='100%')
        )
    
    # 📦 Container
    button_container = widgets.HBox(
        [save_button, reset_button],
        layout=widgets.Layout(
            width=container_width,
            justify_content='flex-end',
            align_items='center',
            margin='5px 0'
        )
    )
    
    # Main container dengan atau tanpa sync info
    if sync_info_widget:
        main_container = widgets.VBox([
            button_container,
            sync_info_widget
        ], layout=widgets.Layout(width=container_width))
    else:
        main_container = button_container
    
    result = {
        'container': main_container,
        'button_container': button_container,
        'save_button': save_button,      # Key konsisten dengan handlers
        'reset_button': reset_button,    # Key konsisten dengan handlers
    }
    
    # Add sync info jika ada
    if sync_info_widget:
        result['sync_info'] = sync_info_widget
    
    return result