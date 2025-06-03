"""
File: smartcash/ui/components/action_buttons.py
Deskripsi: Fixed action buttons dengan one-liner style dan preserved original styling
"""

import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS, ICONS
def create_action_buttons(primary_label: str = "Download Dataset", primary_icon: str = "download", 
                         secondary_buttons: list = None, cleanup_enabled: bool = True,
                         button_width: str = '140px', container_width: str = '100%',
                         primary_style: str = '') -> dict:
    """Create action buttons dengan flexible secondary button support"""
    secondary_buttons = secondary_buttons or [("Check Dataset", "search", "info")]
    
    # Primary button (biasanya download/install)
    download_button = widgets.Button(
        description=primary_label, 
        button_style=primary_style or 'primary', 
        tooltip=f'Klik untuk {primary_label.lower()}',
        icon=primary_icon if primary_icon in ['download', 'save', 'search'] else '',
        layout=widgets.Layout(width=button_width, height='35px')
    )
    setattr(download_button, '_original_style', primary_style or 'primary')
    setattr(download_button, '_original_description', primary_label)
    
    # Secondary button (dari parameter dengan flexible handling)
    if secondary_buttons and len(secondary_buttons) > 0:
        secondary_label, secondary_icon, secondary_style = secondary_buttons[0]
    else:
        secondary_label, secondary_icon, secondary_style = ("Check", "search", "info")
    
    check_button = widgets.Button(
        description=secondary_label,
        button_style=secondary_style,
        tooltip=f'Klik untuk {secondary_label.lower()}',
        icon=secondary_icon if secondary_icon in ['search', 'info', 'check', 'clipboard'] else '',
        layout=widgets.Layout(width=button_width, height='35px')
    )
    setattr(check_button, '_original_style', secondary_style)
    setattr(check_button, '_original_description', secondary_label)
    
    # Optional cleanup/third button
    cleanup_button = None
    if cleanup_enabled:
        cleanup_button = widgets.Button(
            description="Cleanup Dataset",
            button_style='warning',
            tooltip='Hapus dataset yang sudah ada',
            icon='trash',
            layout=widgets.Layout(width=button_width, height='35px')
        )
        setattr(cleanup_button, '_original_style', 'warning')
        setattr(cleanup_button, '_original_description', "Cleanup Dataset")
    
    # Button list untuk container
    button_list = [download_button, check_button]
    if cleanup_button:
        button_list.append(cleanup_button)
    
    # Container dengan proper spacing
    container = widgets.HBox(
        button_list,
        layout=widgets.Layout(
            width=container_width,
            justify_content='flex-start',
            align_items='center',
            margin='10px 0',
            gap='8px'  # Space between buttons
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

def create_save_reset_action_buttons(save_label: str = "Simpan", reset_label: str = "Reset", 
                                   button_width: str = '100px', container_width: str = '100%') -> dict:
    """Create save dan reset buttons dengan one-liner style."""
    save_button = widgets.Button(description=save_label, button_style='primary', tooltip='Simpan konfigurasi saat ini',
                                icon='save', layout=widgets.Layout(width=button_width, height='32px'))
    setattr(save_button, '_original_style', 'primary'), setattr(save_button, '_original_description', save_label)
    
    reset_button = widgets.Button(description=reset_label, button_style='', tooltip='Reset ke nilai default',
                                 icon='refresh', layout=widgets.Layout(width=button_width, height='32px'))
    setattr(reset_button, '_original_style', ''), setattr(reset_button, '_original_description', reset_label)
    
    container = widgets.HBox([save_button, reset_button], layout=widgets.Layout(width=container_width, 
                                                                               justify_content='flex-end', align_items='center', margin='5px 0'))
    return {'container': container, 'save_button': save_button, 'reset_button': reset_button}

def restore_button_original_style(button):
    """Restore button ke original styling dengan one-liner."""
    hasattr(button, '_original_style') and setattr(button, 'button_style', button._original_style)
    hasattr(button, '_original_description') and setattr(button, 'description', button._original_description)
    setattr(button, 'disabled', False)

def set_button_processing_style(button, processing_text: str = "Processing...", processing_style: str = 'warning'):
    """Set button ke processing style dengan one-liner preserve original."""
    not hasattr(button, '_original_style') and setattr(button, '_original_style', button.button_style)
    not hasattr(button, '_original_description') and setattr(button, '_original_description', button.description)
    setattr(button, 'disabled', True), setattr(button, 'description', processing_text), setattr(button, 'button_style', processing_style)
