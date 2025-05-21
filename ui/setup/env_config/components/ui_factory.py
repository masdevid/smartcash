"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: Factory untuk komponen UI environment config
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML, FloatProgress

from smartcash.ui.utils.alert_utils import create_info_box, create_alert
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.utils.layout_utils import STANDARD_LAYOUTS

class UIFactory:
    """
    Factory untuk membuat komponen UI environment config
    """
    
    @staticmethod
    def create_header(title: str, subtitle: str) -> HTML:
        """
        Buat header dengan judul dan subtitle
        
        Args:
            title: Judul header
            subtitle: Subtitle header
            
        Returns:
            Komponen header
        """
        return create_header(title, subtitle)
    
    @staticmethod
    def create_setup_button(description: str = "Konfigurasi Environment", icon: str = "cog") -> Button:
        """
        Buat tombol setup
        
        Args:
            description: Deskripsi tombol
            icon: Ikon tombol
            
        Returns:
            Komponen tombol
        """
        button = Button(
            description=description, 
            button_style="primary",
            icon=icon
        )
        button.layout.width = '100%'
        return button
    
    @staticmethod
    def create_button_layout(button: Button) -> VBox:
        """
        Buat layout untuk tombol
        
        Args:
            button: Tombol yang akan ditempatkan dalam layout
            
        Returns:
            Layout berisi tombol
        """
        return VBox([button], layout=STANDARD_LAYOUTS['vbox'])
    
    @staticmethod
    def create_status_panel(message: str = "Siap untuk mengkonfigurasi environment", 
                           status_type: str = "info") -> widgets.Widget:
        """
        Buat panel status
        
        Args:
            message: Pesan status
            status_type: Tipe status (info, success, error)
            
        Returns:
            Komponen panel status
        """
        return create_status_panel(message, status_type)
    
    @staticmethod
    def create_log_panel(title: str = "Log Konfigurasi Environment") -> Dict[str, widgets.Widget]:
        """
        Buat panel log
        
        Args:
            title: Judul panel log
            
        Returns:
            Dictionary berisi komponen log panel
        """
        return create_log_accordion(title)
    
    @staticmethod
    def create_progress_tracking() -> Dict[str, widgets.Widget]:
        """
        Buat komponen tracking progress
        
        Returns:
            Dictionary berisi komponen progress tracking
        """
        progress_components = create_progress_tracking(module_name="env_config")
        progress_components['progress_message'] = Label(value="")  # Ensure progress_message is included
        return progress_components
        
    @staticmethod
    def create_error_alert(error_message: str) -> widgets.Widget:
        """
        Buat alert error
        
        Args:
            error_message: Pesan error
            
        Returns:
            Alert widget
        """
        return create_alert(f"{ICONS.get('error', '❌')} Error: {error_message}", "danger")
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """
        Buat semua komponen UI untuk environment config
        
        Returns:
            Dictionary berisi semua komponen UI
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
        log_accordion = log_panel['log_accordion']  # Accordion widget for display
        log_output = log_panel['log_output']        # Output widget for logging

        # Progress Bar
        progress_components = cls.create_progress_tracking()

        # Assemble UI components
        ui_components = {
            'header': header,
            'setup_button': setup_button,
            'status_panel': status_panel,
            'log_panel': log_accordion,   # For display
            'log_output': log_output,     # For logging
            'progress_bar': progress_components['progress_bar'],
            'progress_message': progress_components['progress_message'],
            'button_layout': button_layout
        }

        # Create a VBox layout for the entire UI with consistent styling
        ui_layout = VBox([
            header,
            button_layout,
            status_panel,
            log_accordion,  # Use the actual widget
            progress_components['progress_container']
        ], layout=STANDARD_LAYOUTS['vbox'])

        # Add the layout to the components
        ui_components['ui_layout'] = ui_layout

        return ui_components
        
    @classmethod
    def create_error_ui_components(cls, error_message: str = "Terjadi error saat inisialisasi environment") -> Dict[str, Any]:
        """
        Buat komponen UI untuk tampilan error
        
        Args:
            error_message: Pesan error yang akan ditampilkan
            
        Returns:
            Dictionary berisi komponen UI error
        """
        # Header
        header = cls.create_header("Error Konfigurasi Environment", "Terjadi kesalahan saat konfigurasi environment")

        # Error Alert
        error_alert = cls.create_error_alert(error_message)

        # Log Panel
        log_panel = cls.create_log_panel("Log Error")
        log_accordion = log_panel['log_accordion']
        log_output = log_panel['log_output']

        # Assemble UI components
        ui_components = {
            'header': header,
            'error_alert': error_alert,
            'log_panel': log_accordion,
            'log_output': log_output
        }

        # Create a VBox layout for the entire UI
        ui_layout = VBox([
            header,
            error_alert,
            log_accordion
        ], layout=STANDARD_LAYOUTS['vbox'])

        # Add the layout to the components
        ui_components['ui_layout'] = ui_layout

        return ui_components

# Function for backward compatibility
def create_ui_components() -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config (Compatibility function)
    
    Returns:
        Dictionary berisi komponen UI
    """
    return UIFactory.create_ui_components()

def create_error_ui_components() -> Dict[str, Any]:
    """
    Buat komponen UI untuk tampilan error (Compatibility function)
    
    Returns:
        Dictionary berisi komponen UI
    """
    return UIFactory.create_error_ui_components() 