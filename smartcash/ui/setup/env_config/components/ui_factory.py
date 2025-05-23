"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: Factory untuk komponen UI dengan Environment Summary Panel dan layout yang stabil
"""

from typing import Dict, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.setup.env_config.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion

class UIFactory:
    """Factory untuk komponen UI environment config dengan Environment Summary Panel"""
    
    @staticmethod
    def create_setup_button() -> Button:
        """Buat tombol setup dengan styling yang lebih menarik"""
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
        """Panel informasi setup yang informatif"""
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
    
    @staticmethod
    def create_environment_summary_panel() -> HTML:
        """Panel untuk menampilkan Environment Summary yang informatif"""
        return HTML(
            value="""
            <div style="padding: 12px; background-color: #f8f9fa; color: #333; 
                       border-left: 4px solid #17a2b8; border-radius: 4px; margin: 10px 0;
                       font-family: 'Courier New', monospace; font-size: 13px;">
                <div style="text-align: center; color: #6c757d;">
                    🔄 <em>Loading Environment Summary...</em>
                </div>
            </div>
            """
        )
    
    @staticmethod
    def create_requirements_info_panel() -> HTML:
        """Panel informasi requirements dan tips"""
        return HTML(
            value="""
            <div style="padding: 10px; background-color: #fff3cd; color: #856404; 
                       border-left: 4px solid #ffc107; border-radius: 4px; margin: 10px 0;
                       font-size: 12px;">
                <strong>💡 Tips:</strong>
                <ul style="margin: 5px 0; padding-left: 20px; line-height: 1.4;">
                    <li>Setup otomatis akan mount Google Drive jika belum</li>
                    <li>Proses setup membutuhkan koneksi internet yang stabil</li>
                    <li>Symlinks memungkinkan akses data persisten via Drive</li>
                </ul>
            </div>
            """
        )
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """Buat semua komponen UI dengan Environment Summary Panel yang informatif"""
        # Header
        header = create_header(
            "🏗️ Konfigurasi Environment SmartCash", 
            "Setup environment untuk development dan training model dengan info sistem lengkap"
        )

        # Environment Summary Panel (NEW)
        env_summary_panel = cls.create_environment_summary_panel()

        # Info panels
        info_panel = cls.create_info_panel()
        requirements_panel = cls.create_requirements_info_panel()

        # Setup button dengan container
        setup_button = cls.create_setup_button()
        button_container = VBox([setup_button], layout=widgets.Layout(align_items='center'))

        # Status panel
        status_panel = create_status_panel(
            "🔍 Siap untuk konfigurasi environment",
            "info"
        )

        # Log panel dengan height yang lebih kecil untuk accommodate summary
        log_components = create_log_accordion(
            "env_config",
            height='180px',
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

        # Main layout dengan Environment Summary di posisi yang strategis
        ui_layout = VBox([
            header,
            env_summary_panel,  # Environment Summary setelah header
            info_panel,
            requirements_panel,
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
            'env_summary_panel': env_summary_panel,  # NEW component
            'info_panel': info_panel,
            'requirements_panel': requirements_panel,
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