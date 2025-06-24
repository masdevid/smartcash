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
            
            # Import divider dengan fallback
            try:
                from smartcash.ui.components.layout import create_divider
            except ImportError:
                # Fallback divider jika layout module belum tersedia
                import ipywidgets as widgets
                def create_divider():
                    return widgets.HTML(
                        value="<hr style='margin: 15px 0; border: 1px solid #e0e0e0;'>"
                    )
            
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
            status_panel = create_status_panel(
                message="🔍 Siap untuk konfigurasi environment",
                status_type="info"
            )
            
            # Progress tracker dengan error handling
            progress_tracker = None
            try:
                progress_tracker = create_single_progress_tracker(
                    operation="Environment Setup"
                )
                if progress_tracker and hasattr(progress_tracker, 'container'):
                    progress_tracker.container.layout.width = '100%'
            except Exception as e:
                print(f"⚠️ Error creating progress tracker: {e}")
            
            # Log accordion dengan error handling
            log_components = {}
            try:
                log_components = create_log_accordion(
                    module_name="env_config",
                    height='180px',
                    width='100%'
                )
            except Exception as e:
                print(f"⚠️ Error creating log accordion: {e}")
                log_components = {'log_accordion': None, 'log_output': None}
            
            # Main layout dengan flexbox - dua baris layout
            main_layout = cls._create_main_layout(
                header=header,
                env_summary_panel=env_summary_panel,
                info_panel=info_panel,
                button_container=button_container,
                status_panel=status_panel,
                progress_tracker=progress_tracker,
                log_components=log_components
            )
            
            # Return all components
            return {
                # Core UI
                'ui_layout': main_layout,
                'header': header,
                'env_summary_panel': env_summary_panel,
                'info_panel': info_panel,
                'setup_button': setup_button,
                'button_container': button_container,
                'status_panel': status_panel,
                'status_message': status_panel,  # Alias untuk compatibility
                
                # Progress dan logging
                'progress_tracker': progress_tracker,
                'progress_container': progress_tracker.container if progress_tracker else None,
                'log_accordion': log_components.get('log_accordion'),
                'log_output': log_components.get('log_output'),
                
                # Metadata
                'logger_namespace': 'smartcash.ui.env_config',
                'env_config_initialized': True
            }
            
        except Exception as e:
            print(f"🚨 Error creating UI components: {e}")
            return cls._create_fallback_ui(str(e))
    
    @classmethod
    def _create_main_layout(cls, **components) -> widgets.Widget:
        """📐 Create main layout dengan flexbox - dua baris layout"""
        
        # Baris pertama: Summary dan Info (sekarang vertikal)
        summary_section = widgets.VBox([
            widgets.HTML(value="<h3>📊 Environment Summary</h3>"),
            components['env_summary_panel']
        ], layout=widgets.Layout(
            width='100%',
            margin='0px 0px 15px 0px'
        ))
        
        info_section = widgets.VBox([
            widgets.HTML(value="<h3>💡 Tips & Requirements</h3>"),
            components['info_panel']
        ], layout=widgets.Layout(
            width='100%',
            margin='0px 0px 15px 0px'
        ))
        
        # Action section
        action_section = widgets.VBox([
            components['button_container'],
            create_divider(),
            components['status_panel']
        ], layout=widgets.Layout(
            width='100%',
            margin='10px 0px'
        ))
        
        # Progress dan log section
        progress_log_section = widgets.VBox([], layout=widgets.Layout(width='100%'))
        
        if components['progress_tracker']:
            progress_log_section.children = list(progress_log_section.children) + [
                components['progress_tracker'].container
            ]
        
        if components['log_components'].get('log_accordion'):
            progress_log_section.children = list(progress_log_section.children) + [
                components['log_components']['log_accordion']
            ]
        
        # Main container dengan flexbox layout
        main_container = widgets.VBox([
            components['header'],
            create_divider(),
            summary_section,  # Baris pertama
            info_section,     # Baris kedua
            create_divider(),
            action_section,
            progress_log_section
        ], layout=widgets.Layout(
            display='flex',
            flex_direction='column',
            align_items='stretch',
            width='100%',
            max_width='100%',
            overflow='hidden',  # Prevent horizontal scroll
            box_sizing='border-box',
            padding='0px'
        ))
        
        return main_container
    
    @classmethod
    def _create_environment_summary_panel(cls) -> widgets.Widget:
        """📊 Create environment summary panel dengan flexbox"""
        try:
            from smartcash.common.environment import get_environment_info
            
            env_info = get_environment_info()
            
            # Create info cards dengan flexbox
            cards = []
            
            # System info card
            system_info = f"""
            <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 5px 0px; border-left: 4px solid #007bff;">
                <strong>🖥️ System:</strong> {env_info.get('platform', 'Unknown')}<br>
                <strong>🐍 Python:</strong> {env_info.get('python_version', 'Unknown')}<br>
                <strong>📁 Working Dir:</strong> {env_info.get('working_directory', 'Unknown')[:50]}...
            </div>
            """
            cards.append(widgets.HTML(value=system_info))
            
            # GPU info card
            gpu_info = env_info.get('gpu_info', {})
            gpu_available = gpu_info.get('available', False)
            gpu_text = f"""
            <div style="background: {'#d4edda' if gpu_available else '#f8d7da'}; padding: 12px; border-radius: 8px; margin: 5px 0px; border-left: 4px solid {'#28a745' if gpu_available else '#dc3545'};">
                <strong>🎮 GPU:</strong> {'Available' if gpu_available else 'Not Available'}<br>
                <strong>🔧 Device:</strong> {gpu_info.get('device_name', 'CPU Only')}<br>
                <strong>💾 Memory:</strong> {gpu_info.get('memory_info', 'N/A')}
            </div>
            """
            cards.append(widgets.HTML(value=gpu_text))
            
            # Return container dengan flexbox
            return widgets.VBox(cards, layout=widgets.Layout(
                width='100%',
                overflow='hidden'
            ))
            
        except Exception as e:
            return widgets.HTML(value=f"⚠️ Error loading environment info: {e}")
    
    @classmethod
    def _create_info_panel(cls) -> widgets.Widget:
        """💡 Create info panel dengan tips dan requirements"""
        
        tips_html = """
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 5px 0px;">
            <h4 style="margin-top: 0px;">💡 Tips Setup</h4>
            <ul style="margin-bottom: 0px; padding-left: 20px;">
                <li>Pastikan koneksi internet stabil untuk download dependencies</li>
                <li>Proses setup membutuhkan waktu 5-10 menit</li>
                <li>Restart runtime jika diperlukan setelah setup</li>
                <li>Check log output untuk detail progress</li>
            </ul>
        </div>
        """
        
        requirements_html = """
        <div style="background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 5px 0px;">
            <h4 style="margin-top: 0px;">📋 Requirements</h4>
            <ul style="margin-bottom: 0px; padding-left: 20px;">
                <li>Python 3.8+ dengan pip terinstall</li>
                <li>Minimum 2GB RAM untuk training</li>
                <li>Google Drive mounted (untuk Colab)</li>
                <li>CUDA compatible GPU (opsional)</li>
            </ul>
        </div>
        """
        
        return widgets.VBox([
            widgets.HTML(value=tips_html),
            widgets.HTML(value=requirements_html)
        ], layout=widgets.Layout(
            width='100%',
            overflow='hidden'
        ))
    
    @classmethod
    def _create_setup_button(cls) -> widgets.Widget:
        """🔧 Create setup button dengan styling"""
        return widgets.Button(
            description="🚀 Setup Environment",
            button_style='primary',
            tooltip='Klik untuk memulai setup environment SmartCash',
            layout=widgets.Layout(
                width='200px',
                height='40px',
                margin='10px 0px'
            )
        )
    
    @classmethod
    def _create_fallback_ui(cls, error_msg: str) -> Dict[str, Any]:
        """🚨 Create fallback UI jika terjadi error"""
        error_widget = widgets.HTML(
            value=f"""
            <div style="background: #f8d7da; padding: 20px; border-radius: 8px; border: 1px solid #f5c6cb;">
                <h3>🚨 Error Loading UI</h3>
                <p><strong>Error:</strong> {error_msg}</p>
                <p>Silakan refresh cell atau restart runtime.</p>
            </div>
            """
        )
        
        return {
            'ui_layout': error_widget,
            'error': True,
            'error_message': error_msg
        }