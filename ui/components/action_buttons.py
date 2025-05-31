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
    """Create action buttons dengan one-liner style dan preserved original styling."""
    secondary_buttons = secondary_buttons or [("Check Dataset", "search", "info")]
    
    # One-liner button creation dengan preserved styling
    download_button = widgets.Button(description=primary_label, button_style=primary_style, 
                                   tooltip=f'Klik untuk {primary_label.lower()}', icon='download' if primary_icon == 'download' else '',
                                   layout=widgets.Layout(width=button_width, height='35px'))
    setattr(download_button, '_original_style', primary_style), setattr(download_button, '_original_description', primary_label)
    
    check_button = widgets.Button(description="Check Dataset", button_style='info', tooltip='Periksa status dataset yang sudah ada',
                                icon='search', layout=widgets.Layout(width=button_width, height='35px'))
    setattr(check_button, '_original_style', 'info'), setattr(check_button, '_original_description', "Check Dataset")
    
    cleanup_button = (widgets.Button(description="Cleanup Dataset", button_style='warning', tooltip='Hapus dataset yang sudah ada',
                                    icon='trash', layout=widgets.Layout(width=button_width, height='35px')) if cleanup_enabled else None)
    cleanup_button and (setattr(cleanup_button, '_original_style', 'warning'), setattr(cleanup_button, '_original_description', "Cleanup Dataset"))
    
    button_list = [download_button, check_button] + ([cleanup_button] if cleanup_button else [])
    container = widgets.HBox(button_list, layout=widgets.Layout(width=container_width, justify_content='flex-start', 
                                                               align_items='center', margin='10px 0'))
    
    result = {'container': container, 'download_button': download_button, 'check_button': check_button, 'buttons': button_list}
    cleanup_button and result.update({'cleanup_button': cleanup_button})
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
