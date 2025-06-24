"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: UI Factory dengan shared components dan flexbox layout tanpa horizontal scrollbar
"""

import ipywidgets as widgets
from IPython.display import HTML
from typing import Dict, Any, Optional

class UIFactory:
    """🏭 Factory untuk membuat UI components environment config dengan flexbox layout"""
    
    @classmethod
    def create_divider(cls):
        """Create a consistent divider widget"""
        return widgets.HTML(
            value="<hr style='margin: 15px 0; border: 1px solid #e0e0e0;'>"
        )
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """🎨 Create UI components dengan shared components dan flexbox layout"""
        try:
            # Import shared components
            from smartcash.ui.components import (
                create_header,
                create_status_panel,
                create_log_accordion,
                create_single_progress_tracker
            )
            
            # Try to import create_divider from shared components
            try:
                from smartcash.ui.components.layout import create_divider as shared_create_divider
                create_divider = shared_create_divider
            except ImportError:
                # Use local implementation if shared component not available
                create_divider = cls.create_divider
            
            # Core components dengan one-liner creation
            header = create_header(
                title="🏗️ Konfigurasi Environment SmartCash",
                description="Setup environment untuk development dan training model dengan info sistem lengkap"
            )
            
            # Environment summary panel
            env_summary_panel = cls._create_environment_summary_panel()
            
            # Info panel untuk tips dan requirements
            info_panel = cls._create_info_panel()
            
            # Setup button dengan container centered
            setup_button = cls._create_setup_button()
            button_container = widgets.VBox([setup_button], layout=widgets.Layout(
                align_items='center',
                margin='10px 0px',
                width='100%'
            ))
            
            # Status panel
            status_panel = create_status_panel()
            
            # Progress tracker
            progress_tracker = create_single_progress_tracker()
            
            # Log accordion
            log_accordion = create_log_accordion()
            
            # Main layout dengan flexbox
            main_layout = widgets.VBox([
                header,
                create_divider(),
                env_summary_panel,
                info_panel,
                create_divider(),
                button_container,
                status_panel,
                progress_tracker,
                log_accordion
            ], layout=widgets.Layout(
                width='100%',
                max_width='900px',
                margin='0 auto',
                padding='20px',
                box_sizing='border-box'
            ))
            
            return {
                'ui_layout': main_layout,
                'header': header,
                'env_summary_panel': env_summary_panel,
                'info_panel': info_panel,
                'setup_button': setup_button,
                'button_container': button_container,
                'status_panel': status_panel,
                'progress_tracker': progress_tracker,
                'log_accordion': log_accordion,
                'create_divider': create_divider
            }
            
        except ImportError as e:
            print(f"🚨 Import error dalam UIFactory: {e}")
            return cls._create_fallback_ui()
    
    @classmethod
    def _create_environment_summary_panel(cls):
        """🌍 Create environment summary panel"""
        return widgets.HTML(
            value="""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h3>🌍 Environment Summary</h3>
                <div id="env-info">
                    <p>📊 <strong>Platform:</strong> <span id="platform-info">Detecting...</span></p>
                    <p>💾 <strong>Drive Status:</strong> <span id="drive-info">Checking...</span></p>
                    <p>🔧 <strong>Python Version:</strong> <span id="python-info">Loading...</span></p>
                    <p>📁 <strong>Working Directory:</strong> <span id="workdir-info">Getting...</span></p>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%')
        )
    
    @classmethod
    def _create_info_panel(cls):
        """ℹ️ Create info panel dengan tips dan requirements"""
        return widgets.HTML(
            value="""
            <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; 
                       border-left: 4px solid #2196F3; margin: 10px 0;">
                <h4 style="color: #1565C0; margin-top: 0;">💡 Setup Requirements</h4>
                <ul style="color: #424242; margin: 0;">
                    <li>🐍 Python 3.7+ environment</li>
                    <li>📁 Read/write access to working directory</li>
                    <li>🌐 Internet connection for dependencies</li>
                    <li>💾 Minimum 2GB free disk space</li>
                </ul>
            </div>
            """,
            layout=widgets.Layout(width='100%')
        )
    
    @classmethod
    def _create_setup_button(cls):
        """🚀 Create setup button dengan styling"""
        return widgets.Button(
            description='🚀 Setup Environment',
            button_style='primary',
            layout=widgets.Layout(
                width='200px',
                height='45px',
                margin='10px'
            ),
            style=widgets.ButtonStyle(
                font_weight='bold',
                font_size='14px'
            )
        )
    
    @classmethod
    def _create_fallback_ui(cls) -> Dict[str, Any]:
        """🚨 Create fallback UI jika shared components gagal load"""
        fallback_html = widgets.HTML(
            value="""
            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; 
                       border: 1px solid #ffeaa7; margin: 10px 0;">
                <h3>⚠️ Fallback UI Mode</h3>
                <p>Shared components tidak dapat dimuat. Menggunakan basic UI.</p>
                <button onclick="alert('Setup belum tersedia dalam fallback mode')" 
                        style="background: #007bff; color: white; padding: 10px 20px; 
                               border: none; border-radius: 5px; cursor: pointer;">
                    🚀 Setup Environment (Fallback)
                </button>
            </div>
            """
        )
        
        return {
            'ui_layout': fallback_html,
            'fallback_mode': True
        }

# Export UIFactory at module level
__all__ = ['UIFactory']