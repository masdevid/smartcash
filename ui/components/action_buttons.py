"""
File: smartcash/ui/components/action_buttons.py
Deskripsi: Fixed action buttons dengan preserved original styling dan proper button creation
"""

import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS, ICONS

def create_action_buttons(primary_label: str = "Download Dataset",
                         primary_icon: str = "download", 
                         secondary_buttons: list = None,
                         cleanup_enabled: bool = True,
                         button_width: str = '140px',
                         container_width: str = '100%',
                         primary_style: str = '') -> dict:
    """
    Create action buttons dengan preserved original styling dan naming yang konsisten.
    
    Args:
        primary_label: Label untuk primary button
        primary_icon: Icon untuk primary button  
        secondary_buttons: List of (label, icon, style) untuk secondary buttons
        cleanup_enabled: Apakah cleanup button ditampilkan
        button_width: Lebar setiap button
        container_width: Lebar container
        primary_style: Style untuk primary button ('' untuk default, 'success', 'primary', dll)
        
    Returns:
        Dictionary berisi semua button components dengan key yang konsisten
    """
    
    # Default secondary buttons jika tidak disediakan
    if secondary_buttons is None:
        secondary_buttons = [
            ("Check Dataset", "search", "info")
        ]
    
    # 🔘 Primary button dengan preserved styling
    download_button = widgets.Button(
        description=primary_label,
        button_style=primary_style,  # Use provided style, empty string for default
        tooltip=f'Klik untuk {primary_label.lower()}',
        icon='download' if primary_icon == 'download' else '',
        layout=widgets.Layout(width=button_width, height='35px')
    )
    
    # Store original styling untuk restoration
    download_button._original_style = primary_style
    download_button._original_description = primary_label
    
    # 🔍 Check button dengan preserved styling
    check_button = widgets.Button(
        description="Check Dataset",
        button_style='info',
        tooltip='Periksa status dataset yang sudah ada',
        icon='search',
        layout=widgets.Layout(width=button_width, height='35px')
    )
    
    # Store original styling
    check_button._original_style = 'info'
    check_button._original_description = "Check Dataset"
    
    # 🧹 Cleanup button (conditional) dengan preserved styling
    cleanup_button = None
    if cleanup_enabled:
        cleanup_button = widgets.Button(
            description="Cleanup Dataset",
            button_style='warning',
            tooltip='Hapus dataset yang sudah ada',
            icon='trash',
            layout=widgets.Layout(width=button_width, height='35px')
        )
        
        # Store original styling
        cleanup_button._original_style = 'warning'
        cleanup_button._original_description = "Cleanup Dataset"
    
    # 📋 Create button list untuk layout
    button_list = [download_button, check_button]
    if cleanup_button:
        button_list.append(cleanup_button)
    
    # 📦 Container dengan proper spacing
    container = widgets.HBox(
        button_list,
        layout=widgets.Layout(
            width=container_width,
            justify_content='flex-start',
            align_items='center',
            margin='10px 0'
        )
    )
    
    # 📋 Return dictionary dengan key yang konsisten
    result = {
        'container': container,
        'download_button': download_button,  # Key konsisten dengan handlers
        'check_button': check_button,        # Key konsisten dengan handlers  
        'buttons': button_list
    }
    
    # Add cleanup button jika ada
    if cleanup_button:
        result['cleanup_button'] = cleanup_button
    
    return result

def create_save_reset_action_buttons(save_label: str = "Simpan",
                                   reset_label: str = "Reset", 
                                   button_width: str = '100px',
                                   container_width: str = '100%') -> dict:
    """
    Create save dan reset buttons dengan preserved styling dan naming yang konsisten.
    
    Args:
        save_label: Label untuk save button
        reset_label: Label untuk reset button
        button_width: Lebar setiap button
        container_width: Lebar container
        
    Returns:
        Dictionary berisi save dan reset button components
    """
    
    # 💾 Save button dengan preserved styling
    save_button = widgets.Button(
        description=save_label,
        button_style='primary',
        tooltip='Simpan konfigurasi saat ini',
        icon='save',
        layout=widgets.Layout(width=button_width, height='32px')
    )
    
    # Store original styling
    save_button._original_style = 'primary'
    save_button._original_description = save_label
    
    # 🔄 Reset button dengan preserved styling
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',  # Default style
        tooltip='Reset ke nilai default',
        icon='refresh',
        layout=widgets.Layout(width=button_width, height='32px')
    )
    
    # Store original styling
    reset_button._original_style = ''
    reset_button._original_description = reset_label
    
    # 📦 Container
    container = widgets.HBox(
        [save_button, reset_button],
        layout=widgets.Layout(
            width=container_width,
            justify_content='flex-end',
            align_items='center',
            margin='5px 0'
        )
    )
    
    return {
        'container': container,
        'save_button': save_button,      # Key konsisten dengan handlers
        'reset_button': reset_button     # Key konsisten dengan handlers
    }

def restore_button_original_style(button):
    """
    Restore button ke original styling-nya.
    
    Args:
        button: Widget button yang akan di-restore
    """
    if hasattr(button, '_original_style'):
        button.button_style = button._original_style
    if hasattr(button, '_original_description'):
        button.description = button._original_description
    button.disabled = False

def set_button_processing_style(button, processing_text: str = "Processing...", 
                              processing_style: str = 'warning'):
    """
    Set button ke processing style dengan preserve original.
    
    Args:
        button: Widget button
        processing_text: Text saat processing
        processing_style: Style saat processing
    """
    # Store original jika belum ada
    if not hasattr(button, '_original_style'):
        button._original_style = button.button_style
    if not hasattr(button, '_original_description'):
        button._original_description = button.description
    
    # Set processing state
    button.disabled = True
    button.description = processing_text
    button.button_style = processing_style