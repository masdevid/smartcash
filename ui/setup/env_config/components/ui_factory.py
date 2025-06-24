"""
File: smartcash/ui/setup/env_config/components/ui_factory.py
Deskripsi: UI Factory dengan constants integration dan clean structure
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML
import logging

# Import fallback utilities
try:
    from smartcash.ui.utils.fallback_utils import (
        try_operation_safe,
        create_fallback_ui,
        create_minimal_ui,
        get_safe_logger,
        FallbackConfig
    )
    from smartcash.ui.utils.constants import ICONS, COLORS
    HAS_FALLBACK_UTILS = True
except ImportError as e:
    print(f"Warning: Fallback utilities not available: {e}")
    HAS_FALLBACK_UTILS = False

# Setup logger
logger = get_safe_logger('smartcash.ui.setup.env_config')

# Try to import UI components with fallbacks
def safe_import(component_name: str, fallback: Callable = None) -> Callable:
    """Safely import a UI component with fallback"""
    try:
        from smartcash.ui.components import __dict__ as components_dict
        component = components_dict.get(f"create_{component_name}")
        return component or fallback or (lambda *a, **kw: HTML(f"<div>⚠️ Component {component_name} not found</div>"))
    except ImportError as e:
        logger.warning(f"Failed to import {component_name}: {e}")
        return fallback or (lambda *a, **kw: HTML(f"<div>⚠️ Failed to load {component_name}</div>"))

# Import components with fallbacks
create_header = safe_import('header')
create_status_panel = safe_import('status_panel')
create_single_progress_tracker = safe_import('single_progress_tracker')
create_log_accordion = safe_import('log_accordion')

from smartcash.ui.setup.env_config.constants import CONFIG_TEMPLATES

# Get icon helper function
def get_icon(icon_name: str, default: str = "ℹ️") -> str:
    """Get icon from ICONS constant with fallback"""
    if not HAS_FALLBACK_UTILS or not ICONS:
        return default
    return ICONS.get(icon_name, default)

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
    
    @classmethod
    def create_environment_summary_panel(cls) -> HBox:
        """Environment summary panel dengan loading state dalam layout kolom"""
        summary_panel = HTML(value="""
            <div style="padding: 12px; background-color: #f8f9fa; color: #333; 
                       border-left: 4px solid #17a2b8; border-radius: 4px; margin: 0 5px;
                       font-family: 'Courier New', monospace; font-size: 13px; height: 100%;">
                <div style="text-align: center; color: #6c757d;">
                    🔄 <em>Loading Environment Summary...</em>
                </div>
            </div>
            """)
        
        requirements_panel = cls._create_requirements_info_panel()
        
        # Buat layout dua kolom dengan HBox
        return HBox(
            [summary_panel, requirements_panel],
            layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                margin='10px 0',
                padding='0 5px'
            )
        )
    
    @classmethod
    def _create_requirements_info_panel(cls) -> HTML:
        """Internal method untuk requirements panel"""
        essential_configs = ['base_config', 'model_config', 'training_config', 'backbone_config', 'dataset_config']
        return HTML(value=f"""
            <div style="padding: 12px; background-color: #fff3cd; color: #856404; 
                       border-left: 4px solid #ffc107; border-radius: 4px; margin: 0 5px;
                       font-size: 13px; height: 100%;">
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
    def create_requirements_info_panel(cls) -> HTML:
        """Create requirements info panel with loading state"""
        return HTML(value="""
            <div style="padding: 12px; background-color: #f8f9fa; color: #333; 
                       border-left: 4px solid #28a745; border-radius: 4px; margin: 0 5px;
                       font-family: 'Courier New', monospace; font-size: 13px; height: 100%;">
                <div style="text-align: center; color: #6c757d;">
                    📦 <em>Checking Requirements...</em>
                </div>
            </div>
            """)
    
    @classmethod
    def create_ui_components(cls) -> Dict[str, Any]:
        """Create UI components dengan constants integration"""
        try:
            # Core components dengan one-liner creation
            header = create_header(
                title="🏗️ Konfigurasi Environment SmartCash",
                description="Setup environment untuk development dan training model dengan info sistem lengkap"
            )
            env_summary_panel = cls.create_environment_summary_panel()
            info_panel = cls.create_info_panel()
            requirements_panel = cls.create_requirements_info_panel()
            
            # Setup button dengan container centered
            setup_button = cls.create_setup_button()
            button_container = VBox([setup_button], layout=widgets.Layout(
                align_items='center',
                margin='10px 0'
            ))
            
            # Status dan log components
            status_panel = create_status_panel(
                message="🔍 Siap untuk konfigurasi environment",
                status_type="info"
            )
            
            # Inisialisasi log components dengan error handling
            try:
                log_components = create_log_accordion(
                    module_name="env_config",
                    height='180px',
                    width='100%'
                )
            except Exception as e:
                print(f"Error creating log accordion: {e}")
                log_components = {'log_accordion': None, 'log_output': None}
            
            # Inisialisasi progress tracker dengan error handling
            progress_tracker = None
            progress_container = None
            try:
                from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
                
                progress_tracker = create_single_progress_tracker(
                    operation="Environment Setup"
                )
                progress_container = progress_tracker.container
                if progress_container:
                    progress_container.layout.width = '100%'
                    progress_container.layout.visibility = 'visible'
            except Exception as e:
                print(f"Error initializing progress tracker: {e}")
                progress_container = VBox(layout=widgets.Layout(visibility='hidden'))
                
            # Main layout dengan urutan optimal dan error handling
            try:
                ui_children = [
                    header, 
                    button_container, 
                    status_panel,
                    progress_container if progress_container is not None else VBox(layout=widgets.Layout(visibility='hidden')),
                ]
                
                # Tambahkan log accordion jika tersedia
                if log_components and 'log_accordion' in log_components and log_components['log_accordion'] is not None:
                    ui_children.append(log_components['log_accordion'])
                    
                # Tambahkan komponen tambahan
                ui_children.extend([
                    env_summary_panel,
                    requirements_panel
                ])
                
                ui_layout = VBox(
                    ui_children,
                    layout=widgets.Layout(
                        width='100%',
                        padding='15px',
                        border='1px solid #ddd',
                        border_radius='8px',
                        background_color='#fafafa'
                    )
                )
                
                # Siapkan komponen untuk return
                components = {
                    'header': header,
                    'status_panel': status_panel,
                    'env_summary_panel': env_summary_panel,
                    'requirements_panel': requirements_panel,
                    'setup_button': setup_button,
                    'button_container': button_container,
                    'ui_layout': ui_layout
                }
                
                # Tambahkan komponen opsional jika tersedia
                if log_components:
                    components.update({
                        'log_accordion': log_components.get('log_accordion'),
                        'log_output': log_components.get('log_output')
                    })
                    
                if progress_tracker is not None:
                    components.update({
                        'progress_tracker': progress_tracker,
                        'progress_bar': progress_tracker.progress_bars.get('main') if hasattr(progress_tracker, 'progress_bars') else None,
                        'progress_message': progress_tracker.status_widget if hasattr(progress_tracker, 'status_widget') else None,
                        'progress_container': progress_container
                    })
                    
                return components
                
            except Exception as e:
                print(f"Error creating UI layout: {e}")
                # Fallback layout minimal
                return {
                    'header': header,
                    'status_panel': status_panel,
                    'setup_button': setup_button,
                    'button_container': button_container,
                    'ui_layout': VBox([
                        header,
                        button_container,
                        status_panel,
                        HTML("<div style='color: red;'>Error initializing UI components</div>")
                    ])
                }
                
        except Exception as e:
            print(f"Error initializing main components: {e}")
            # Kembalikan UI minimal dengan pesan error
            error_msg = f"Gagal memuat komponen UI: {str(e)}"
            return {
                'ui_layout': VBox([
                    HTML(f"<div style='color: red; padding: 20px;'>{error_msg}</div>"),
                    HTML("<div>Silakan refresh halaman atau hubungi tim pengembang.</div>")
                ]),
                'error': str(e)
            }