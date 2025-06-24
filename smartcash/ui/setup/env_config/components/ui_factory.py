# File: ui/setup/env_config/components/ui_factory.py
# Deskripsi: UI Factory yang menggunakan shared components dengan benar

import ipywidgets as widgets
from typing import Dict, Any

class UIFactory:
    """🏭 Factory untuk environment config UI menggunakan shared components"""
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """🎨 Create UI components menggunakan shared components sepenuhnya"""
        try:
            # Import shared components - INI YANG BENAR
            from smartcash.ui.components import (
                create_header,
                create_status_panel, 
                create_log_accordion,
                create_single_progress_tracker,
                create_divider
            )
            
            # Core components dengan shared functions
            header = create_header(
                title="🏗️ Konfigurasi Environment SmartCash",
                description="Setup environment untuk development dan training model"
            )
            
            # Status panel menggunakan shared component
            status_panel = create_status_panel()
            
            # Setup button - HANYA button, bukan UI creation
            setup_button = widgets.Button(
                description='🚀 Setup Environment',
                button_style='primary',
                layout=widgets.Layout(width='200px', height='45px')
            )
            
            # Button container
            button_container = widgets.VBox([setup_button], 
                layout=widgets.Layout(align_items='center', margin='10px 0px'))
            
            # Progress tracker menggunakan shared component
            progress_tracker = create_single_progress_tracker()
            
            # Log accordion menggunakan shared component  
            log_accordion = create_log_accordion()
            
            # Environment info panel - simple HTML widget
            env_info_panel = widgets.HTML(
                value="""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>🌍 Environment Info</h4>
                    <p id="env-status">🔍 Checking environment status...</p>
                </div>
                """
            )
            
            # Requirements info panel
            requirements_panel = widgets.HTML(
                value="""
                <div style="background: #e8f4fd; padding: 12px; border-radius: 6px; 
                           border-left: 3px solid #2196F3; margin: 10px 0;">
                    <h5 style="color: #1565C0; margin-top: 0;">💡 Requirements</h5>
                    <ul style="color: #424242; margin: 5px 0;">
                        <li>🐍 Python 3.7+ environment</li>
                        <li>📁 Read/write directory access</li>
                        <li>🌐 Internet connection</li>
                    </ul>
                </div>
                """
            )
            
            # Main layout - menggunakan shared components
            main_layout = widgets.VBox([
                header,
                create_divider(),
                env_info_panel,
                requirements_panel,
                create_divider(),
                button_container,
                status_panel,
                progress_tracker,  # Shared component sudah return widget
                log_accordion     # Shared component sudah return widget
            ], layout=widgets.Layout(
                width='100%',
                max_width='800px',
                margin='0 auto',
                padding='20px'
            ))
            
            # Return dictionary dengan proper widget references
            return {
                'ui_layout': main_layout,
                'header': header,
                'env_info_panel': env_info_panel,
                'requirements_panel': requirements_panel, 
                'setup_button': setup_button,
                'button_container': button_container,
                'status_panel': status_panel,
                'progress_tracker': progress_tracker,
                'log_accordion': log_accordion,
                # Metadata
                'initialized': True,
                'module_name': 'environment_config'
            }
            
        except ImportError as e:
            print(f"🚨 Import error shared components: {e}")
            return cls._create_minimal_fallback()
    
    @classmethod
    def _create_minimal_fallback(cls) -> Dict[str, Any]:
        """🚨 Minimal fallback tanpa dependencies"""
        
        fallback_widget = widgets.HTML(
            value="""
            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; 
                       border: 1px solid #ffeaa7; margin: 10px 0;">
                <h3>⚠️ Environment Config - Fallback Mode</h3>
                <p>Shared components tidak dapat dimuat. Gunakan fallback mode.</p>
                <p><strong>Solusi:</strong></p>
                <ol>
                    <li>Restart runtime: <code>Runtime → Restart runtime</code></li>
                    <li>Check imports: Pastikan shared components tersedia</li>
                </ol>
            </div>
            """
        )
        
        return {
            'ui_layout': fallback_widget,
            'fallback_mode': True,
            'initialized': False,
            'module_name': 'environment_config_fallback'
        }

# Export factory class
__all__ = ['UIFactory']