"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: Factory untuk komponen UI environment config - diperbaiki untuk menghilangkan horizontal scrollbar dan memisahkan business logic
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion

class UIFactory:
    """
    Factory untuk membuat komponen UI environment config - fokus hanya pada UI tanpa business logic
    """
    
    @staticmethod
    def create_header(title: str, subtitle: str) -> HTML:
        """Buat header dengan judul dan subtitle"""
        return create_header(title, subtitle)
    
    @staticmethod
    def create_setup_button(description: str = "Konfigurasi Environment", icon: str = "cog") -> Button:
        """Buat tombol setup dengan layout yang proper"""
        return Button(
            description=description, 
            button_style="primary",
            icon=icon,
            layout=widgets.Layout(
                width='auto',  # Changed from '100%' to 'auto'
                min_width='200px',
                margin='5px 0'
            )
        )
    
    @staticmethod
    def create_button_layout(button: Button) -> VBox:
        """Buat layout untuk tombol dengan proper spacing"""
        return VBox(
            [button], 
            layout=widgets.Layout(
                align_items='center',  # Center the button
                width='100%',
                margin='10px 0'
            )
        )
    
    @staticmethod
    def create_status_panel(message: str = "Siap untuk mengkonfigurasi environment", 
                           status_type: str = "info") -> widgets.HTML:
        """Buat panel status dengan layout yang tidak overflow"""
        return create_status_panel(
            message, 
            status_type,
            layout={'max_width': '100%', 'overflow': 'hidden'}  # Prevent overflow
        )
    
    @staticmethod
    def create_log_panel(title: str = "Log Konfigurasi Environment") -> Dict[str, widgets.Widget]:
        """Buat panel log dengan height yang terbatas"""
        return create_log_accordion(
            title.replace("Log ", "").lower(),  # Extract module name
            height='150px',  # Fixed height to prevent overflow
            width='100%'
        )
    
    @staticmethod
    def create_progress_tracking() -> Dict[str, widgets.Widget]:
        """Buat komponen tracking progress dengan layout yang compact"""
        progress_components = create_progress_tracking(
            module_name="env_config",
            width='100%'
        )
        
        # Ensure progress_message exists
        if 'progress_message' not in progress_components:
            progress_components['progress_message'] = Label(
                value="", 
                layout=widgets.Layout(max_width='100%', overflow='hidden')
            )
        
        return progress_components
        
    @staticmethod
    def create_error_alert(error_message: str) -> widgets.HTML:
        """Buat alert error dengan text wrapping"""
        return widgets.HTML(
            value=f"""
            <div style="padding: 10px; background-color: #f8d7da; color: #721c24; 
                       border-left: 4px solid #dc3545; border-radius: 5px; margin: 10px 0;
                       word-wrap: break-word; max-width: 100%;">
                <h4 style="margin-top: 0; color: inherit;">{ICONS.get('error', '❌')} Error</h4>
                <p style="word-wrap: break-word;">{error_message}</p>
            </div>
            """,
            layout=widgets.Layout(max_width='100%', overflow='hidden')
        )
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """
        Buat semua komponen UI untuk environment config dengan layout yang responsive
        """
        # Header
        header = cls.create_header("Konfigurasi Environment", "Atur dan konfigurasi environment untuk SmartCash")

        # Setup button
        setup_button = cls.create_setup_button()
        button_layout = cls.create_button_layout(setup_button)

        # Status Panel
        status_panel = cls.create_status_panel()

        # Log Panel
        log_panel = cls.create_log_panel()
        log_accordion = log_panel['log_accordion']
        log_output = log_panel['log_output']

        # Progress Bar
        progress_components = cls.create_progress_tracking()

        # Assemble UI components
        ui_components = {
            'header': header,
            'setup_button': setup_button,
            'status_panel': status_panel,
            'log_panel': log_accordion,
            'log_output': log_output,
            'progress_bar': progress_components['progress_bar'],
            'progress_message': progress_components.get('progress_message'),
            'button_layout': button_layout
        }

        # Create main layout with proper spacing and no overflow
        ui_layout = VBox([
            header,
            button_layout,
            status_panel,
            log_accordion,
            progress_components['progress_container']
        ], layout=widgets.Layout(
            width='100%',
            max_width='100%',
            overflow='hidden',  # Prevent horizontal scrollbar
            margin='0',
            padding='10px'
        ))

        ui_components['ui_layout'] = ui_layout
        return ui_components
        
    @classmethod
    def create_error_ui_components(cls, error_message: str = "Terjadi error saat inisialisasi environment") -> Dict[str, Any]:
        """Buat komponen UI untuk tampilan error dengan layout yang proper"""
        # Header
        header = cls.create_header("Error Konfigurasi Environment", "Terjadi kesalahan saat konfigurasi environment")

        # Error Alert
        error_alert = cls.create_error_alert(error_message)

        # Log Panel
        log_panel = cls.create_log_panel("Error")
        log_accordion = log_panel['log_accordion']
        log_output = log_panel['log_output']

        # Assemble UI components
        ui_components = {
            'header': header,
            'error_alert': error_alert,
            'log_panel': log_accordion,
            'log_output': log_output
        }

        # Create layout with proper overflow handling
        ui_layout = VBox([
            header,
            error_alert,
            log_accordion
        ], layout=widgets.Layout(
            width='100%',
            max_width='100%',
            overflow='hidden',
            margin='0',
            padding='10px'
        ))

        ui_components['ui_layout'] = ui_layout
        return ui_components


# Backward compatibility functions
def create_ui_components() -> Dict[str, Any]:
    """Buat komponen UI untuk environment config (Compatibility function)"""
    return UIFactory.create_ui_components()

def create_error_ui_components(error_message: str = "Terjadi error saat inisialisasi environment") -> Dict[str, Any]:
    """Buat komponen UI untuk tampilan error (Compatibility function)"""
    return UIFactory.create_error_ui_components(error_message)