"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: UI Factory dengan constants integration dan clean structure
"""

from typing import Dict, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.progress_tracker import create_single_progress_tracker
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.setup.env_config.constants import CONFIG_TEMPLATES

class UIFactory:
    """UI Factory dengan constants integration untuk consistent styling"""
    
    @staticmethod
    def create_setup_button() -> Button:
        """Setup button dengan standard styling - one-liner"""
        return Button(description="🔧 Konfigurasi Environment", button_style="primary", icon="cog",
                     layout=widgets.Layout(width='auto', min_width='250px', margin='10px 0'))
    
    @staticmethod
    def create_info_panel() -> HTML:
        """Info panel dengan updated config count"""
        config_count = len(CONFIG_TEMPLATES)
        return HTML(value=f"""
            <div style="padding: 12px; background-color: #e3f2fd; color: #1565c0; 
                       border-left: 4px solid #2196f3; border-radius: 4px; margin: 10px 0;">
                <h4 style="margin-top: 0;">🏗️ Setup Environment SmartCash</h4>
                <ul style="margin: 8px 0; padding-left: 20px; line-height: 1.5;">
                    <li>📁 Membuat direktori di Drive</li>
                    <li>📋 Clone {config_count} config templates</li>
                    <li>🔗 Setup symlinks</li>
                    <li>🔧 Inisialisasi managers</li>
                </ul>
            </div>
            """)
    
    @staticmethod
    def create_environment_summary_panel() -> HTML:
        """Environment summary panel dengan loading state"""
        return HTML(value="""
            <div style="padding: 12px; background-color: #f8f9fa; color: #333; 
                       border-left: 4px solid #17a2b8; border-radius: 4px; margin: 10px 0;
                       font-family: 'Courier New', monospace; font-size: 13px;">
                <div style="text-align: center; color: #6c757d;">
                    🔄 <em>Loading Environment Summary...</em>
                </div>
            </div>
            """)
    
    @staticmethod
    def create_requirements_info_panel() -> HTML:
        """Requirements panel dengan config templates info"""
        essential_configs = ['base_config', 'model_config', 'training_config', 'backbone_config', 'dataset_config']
        return HTML(value=f"""
            <div style="padding: 10px; background-color: #fff3cd; color: #856404; 
                       border-left: 4px solid #ffc107; border-radius: 4px; margin: 10px 0;
                       font-size: 12px;">
                <strong>💡 Tips Setup Environment:</strong>
                <ul style="margin: 5px 0; padding-left: 20px; line-height: 1.4;">
                    <li>🔗 Setup otomatis akan mount Google Drive jika belum</li>
                    <li>📁 Symlinks memungkinkan data tersimpan persisten di Drive</li>
                    <li>⚡ Environment siap untuk development dan training model</li>
                    <li>📋 Essential configs: {', '.join(essential_configs)}</li>
                </ul>
            </div>
            """)
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """Create UI components dengan constants integration"""
        # Core components dengan one-liner creation
        header = create_header("🏗️ Konfigurasi Environment SmartCash", 
                              "Setup environment untuk development dan training model dengan info sistem lengkap")
        env_summary_panel = cls.create_environment_summary_panel()
        info_panel = cls.create_info_panel()
        requirements_panel = cls.create_requirements_info_panel()
        
        # Setup button dengan container centered
        setup_button = cls.create_setup_button()
        button_container = VBox([setup_button], layout=widgets.Layout(align_items='center'))
        
        # Status dan log components
        status_panel = create_status_panel("🔍 Siap untuk konfigurasi environment", "info")
        log_components = create_log_accordion("env_config", height='180px', width='100%')
        
        # Progress tracker dengan proper integration
        progress_tracker = create_single_progress_tracker(operation="Environment Setup")
        progress_container = progress_tracker.container
        progress_container.layout.width = '100%'
        progress_container.layout.visibility = 'visible'
        
        # Main layout dengan urutan optimal
        ui_layout = VBox([
            header, button_container, status_panel, progress_container,
            log_components['log_accordion'], env_summary_panel, requirements_panel
        ], layout=widgets.Layout(width='100%', padding='15px', border='1px solid #ddd',
                                border_radius='8px', background_color='#fafafa'))
        
        return {
            'header': header, 'status_panel': status_panel, 'env_summary_panel': env_summary_panel,
            'requirements_panel': requirements_panel, 'setup_button': setup_button,
            'button_container': button_container, 'log_accordion': log_components['log_accordion'],
            'log_output': log_components['log_output'], 'progress_tracker': progress_tracker,
            'progress_bar': progress_tracker.progress_bars.get('main'),
            'progress_message': progress_tracker.status_widget,
            'progress_container': progress_container, 'ui_layout': ui_layout
        }