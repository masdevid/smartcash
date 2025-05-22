"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: Factory untuk komponen UI environment config - diperbaiki dengan status yang lebih informatif
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.setup.env_config.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion

class UIFactory:
    """
    Factory untuk membuat komponen UI environment config dengan layout yang responsive
    """
    
    @staticmethod
    def create_header(title: str, subtitle: str) -> HTML:
        """Buat header dengan judul dan subtitle"""
        return create_header(title, subtitle)
    
    @staticmethod
    def create_setup_button(description: str = "🔧 Konfigurasi Environment", icon: str = "cog") -> Button:
        """Buat tombol setup dengan styling yang menarik"""
        return Button(
            description=description, 
            button_style="primary",
            icon=icon,
            layout=widgets.Layout(
                width='auto',
                min_width='250px',
                margin='10px 0',
                border='2px solid #0066cc',
                font_weight='bold'
            ),
            style={'button_color': '#0066cc', 'font_weight': 'bold'}
        )
    
    @staticmethod
    def create_button_layout(button: Button) -> VBox:
        """Buat layout untuk tombol dengan proper spacing"""
        help_text = HTML(
            value="""
            <div style="text-align: center; margin: 5px 0; color: #666; font-size: 12px;">
                💡 <em>Setup akan membuat direktori, konfigurasi, dan symlinks yang diperlukan</em>
            </div>
            """
        )
        
        return VBox(
            [button, help_text], 
            layout=widgets.Layout(
                align_items='center',
                width='100%',
                margin='10px 0'
            )
        )
    
    @staticmethod
    def create_status_panel(message: str = "🔍 Siap untuk mengkonfigurasi environment", 
                           status_type: str = "info") -> widgets.HTML:
        """Buat panel status dengan styling yang informatif"""
        return create_status_panel(
            message, 
            status_type,
            layout={'max_width': '100%', 'overflow': 'hidden', 'margin': '10px 0'}
        )
    
    @staticmethod
    def create_log_panel(title: str = "Log Konfigurasi Environment") -> Dict[str, widgets.Widget]:
        """Buat panel log dengan height yang optimal"""
        log_components = create_log_accordion(
            title.replace("Log ", "").lower(),
            height='200px',  # Increased height for better readability
            width='100%'
        )
        
        # Add helpful description
        log_components['description'] = HTML(
            value="""
            <div style="margin: 5px 0; color: #666; font-size: 11px;">
                📋 <em>Detail proses konfigurasi akan ditampilkan di sini</em>
            </div>
            """
        )
        
        return log_components
    
    @staticmethod
    def create_progress_tracking() -> Dict[str, widgets.Widget]:
        """Buat komponen tracking progress dengan layout yang informatif"""
        progress_components = create_progress_tracking(
            module_name="env_config",
            width='100%'
        )
        
        # Ensure progress_message exists dengan styling
        if 'progress_message' not in progress_components:
            progress_components['progress_message'] = Label(
                value="", 
                layout=widgets.Layout(
                    max_width='100%', 
                    overflow='hidden',
                    margin='5px 0'
                ),
                style={'font_size': '12px', 'text_color': '#666'}
            )
        
        return progress_components
        
    @staticmethod
    def create_error_alert(error_message: str) -> widgets.HTML:
        """Buat alert error dengan styling yang menarik"""
        return widgets.HTML(
            value=f"""
            <div style="padding: 15px; background-color: #fff3cd; color: #856404; 
                       border: 1px solid #ffeaa7; border-radius: 8px; margin: 10px 0;
                       word-wrap: break-word; max-width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #721c24; display: flex; align-items: center;">
                    {ICONS.get('error', '❌')} <span style="margin-left: 8px;">Error Setup Environment</span>
                </h4>
                <p style="word-wrap: break-word; margin-bottom: 0; line-height: 1.4;">{error_message}</p>
                <div style="margin-top: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px; font-size: 12px;">
                    💡 <strong>Tip:</strong> Coba refresh browser dan jalankan setup lagi
                </div>
            </div>
            """,
            layout=widgets.Layout(max_width='100%', overflow='hidden')
        )
    
    @staticmethod
    def create_info_panel() -> widgets.HTML:
        """Buat panel informasi untuk environment setup"""
        return widgets.HTML(
            value="""
            <div style="padding: 12px; background-color: #e3f2fd; color: #1565c0; 
                       border-left: 4px solid #2196f3; border-radius: 4px; margin: 10px 0;">
                <h4 style="margin-top: 0; display: flex; align-items: center;">
                    🏗️ <span style="margin-left: 8px;">Setup Environment SmartCash</span>
                </h4>
                <ul style="margin: 8px 0; padding-left: 20px; line-height: 1.5;">
                    <li>📁 Membuat direktori: data, exports, logs, models, output</li>
                    <li>🔗 Setup symlinks ke Google Drive (jika Colab)</li>
                    <li>📝 Menyalin file konfigurasi default</li>
                    <li>🔧 Inisialisasi config manager</li>
                </ul>
            </div>
            """
        )
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """
        Buat semua komponen UI untuk environment config dengan layout yang responsive dan informatif
        """
        # Header
        header = cls.create_header("🏗️ Konfigurasi Environment SmartCash", 
                                  "Setup dan konfigurasi environment untuk development dan training model")

        # Info panel
        info_panel = cls.create_info_panel()

        # Setup button
        setup_button = cls.create_setup_button()
        button_layout = cls.create_button_layout(setup_button)

        # Status Panel
        status_panel = cls.create_status_panel()

        # Log Panel
        log_panel = cls.create_log_panel()
        log_accordion = log_panel['log_accordion']
        log_output = log_panel['log_output']
        log_description = log_panel.get('description')

        # Progress Bar
        progress_components = cls.create_progress_tracking()

        # Assemble UI components
        ui_components = {
            'header': header,
            'info_panel': info_panel,
            'setup_button': setup_button,
            'status_panel': status_panel,
            'log_panel': log_accordion,
            'log_output': log_output,
            'progress_bar': progress_components['progress_bar'],
            'progress_message': progress_components.get('progress_message'),
            'button_layout': button_layout
        }

        # Create main layout dengan spacing yang optimal
        main_components = [header, info_panel, button_layout, status_panel]
        
        # Add log description if exists
        if log_description:
            main_components.append(log_description)
            
        main_components.extend([
            log_accordion,
            progress_components['progress_container']
        ])

        ui_layout = VBox(main_components, layout=widgets.Layout(
            width='100%',
            max_width='100%',
            overflow='hidden',
            margin='0',
            padding='15px',
            border='1px solid #ddd',
            border_radius='8px',
            background_color='#fafafa'
        ))

        ui_components['ui_layout'] = ui_layout
        return ui_components
        
    @classmethod
    def create_error_ui_components(cls, error_message: str = "Terjadi error saat inisialisasi environment") -> Dict[str, Any]:
        """Buat komponen UI untuk tampilan error dengan styling yang informatif"""
        # Header
        header = cls.create_header("❌ Error Konfigurasi Environment", 
                                  "Terjadi kesalahan saat konfigurasi environment")

        # Error Alert
        error_alert = cls.create_error_alert(error_message)

        # Log Panel
        log_panel = cls.create_log_panel("Error Log")
        log_accordion = log_panel['log_accordion']
        log_output = log_panel['log_output']

        # Retry button
        retry_button = Button(
            description="🔄 Coba Lagi",
            button_style="warning",
            layout=widgets.Layout(width='auto', min_width='150px', margin='10px 0')
        )
        
        retry_layout = VBox([retry_button], layout=widgets.Layout(align_items='center'))

        # Assemble UI components
        ui_components = {
            'header': header,
            'error_alert': error_alert,
            'retry_button': retry_button,
            'log_panel': log_accordion,
            'log_output': log_output
        }

        # Create layout dengan proper error handling
        ui_layout = VBox([
            header,
            error_alert,
            retry_layout,
            log_accordion
        ], layout=widgets.Layout(
            width='100%',
            max_width='100%',
            overflow='hidden',
            margin='0',
            padding='15px',
            border='1px solid #dc3545',
            border_radius='8px',
            background_color='#fff5f5'
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