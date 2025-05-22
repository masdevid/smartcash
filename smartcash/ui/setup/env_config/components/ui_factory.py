"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: Factory untuk komponen UI dengan progress yang bisa disembunyikan dan layout yang stabil
"""

from typing import Dict, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.setup.env_config.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion

class UIFactory:
    """Factory untuk komponen UI environment config yang stabil"""
    
    @staticmethod
    def create_setup_button() -> Button:
        """Buat tombol setup"""
        return Button(
            description="🔧 Konfigurasi Environment",
            button_style="primary",
            icon="cog",
            layout=widgets.Layout(
                width='auto',
                min_width='250px',
                margin='10px 0'
            )
        )
    
    @staticmethod
    def create_info_panel() -> HTML:
        """Panel informasi setup"""
        return HTML(
            value="""
            <div style="padding: 12px; background-color: #e3f2fd; color: #1565c0; 
                       border-left: 4px solid #2196f3; border-radius: 4px; margin: 10px 0;">
                <h4 style="margin-top: 0;">🏗️ Setup Environment SmartCash</h4>
                <ul style="margin: 8px 0; padding-left: 20px; line-height: 1.5;">
                    <li>📁 Membuat direktori di Drive</li>
                    <li>📋 Clone config templates</li>
                    <li>🔗 Setup symlinks</li>
                    <li>🔧 Inisialisasi managers</li>
                </ul>
            </div>
            """
        )
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """Buat semua komponen UI dengan visibility control untuk progress"""
        # Header
        header = create_header(
            "🏗️ Konfigurasi Environment SmartCash", 
            "Setup environment untuk development dan training model"
        )

        # Info panel
        info_panel = cls.create_info_panel()

        # Setup button dengan container
        setup_button = cls.create_setup_button()
        button_container = VBox([setup_button], layout=widgets.Layout(align_items='center'))

        # Status panel
        status_panel = create_status_panel(
            "🔍 Siap untuk konfigurasi environment",
            "info"
        )

        # Log panel
        log_components = create_log_accordion(
            "env_config",
            height='200px',
            width='100%'
        )

        # Progress tracking dengan visibility control
        progress_components = create_progress_tracking(
            module_name="env_config",
            width='100%'
        )
        
        # Progress container yang bisa disembunyikan
        progress_container = VBox([
            progress_components['progress_bar'],
            progress_components['progress_message']
        ], layout=widgets.Layout(
            width='100%',
            visibility='visible'  # Default visible
        ))

        # Main layout
        ui_layout = VBox([
            header,
            info_panel,
            button_container,
            status_panel,
            log_components['log_accordion'],
            progress_container
        ], layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #ddd',
            border_radius='8px',
            background_color='#fafafa'
        ))

        return {
            'header': header,
            'info_panel': info_panel,
            'setup_button': setup_button,
            'button_container': button_container,
            'status_panel': status_panel,
            'log_accordion': log_components['log_accordion'],
            'log_output': log_components['log_output'],
            'progress_bar': progress_components['progress_bar'],
            'progress_message': progress_components['progress_message'],
            'progress_container': progress_container,
            'ui_layout': ui_layout
        }