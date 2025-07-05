"""
file_path: smartcash/ui/setup/colab/components/colab_ui.py
Deskripsi: Komponen UI untuk environment configuration dengan layout lengkap menggunakan container baru
"""

from typing import Dict, Any, Optional, List

# Import container components
from smartcash.ui.components.main_container import (
    create_main_container,
    ContainerConfig
)
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container, PanelConfig, PanelType
from smartcash.ui.components.form_container import create_form_container, LayoutType

# Import local colab components
from smartcash.ui.setup.colab.components.setup_summary import create_setup_summary
from smartcash.ui.setup.colab.components.env_info_panel import create_env_info_panel
from smartcash.ui.setup.colab.components.tips_panel import create_tips_requirements

def create_colab_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi Colab dengan container baru
    
    Args:
        config: Konfigurasi untuk UI components
        **kwargs: Additional arguments
        
    Returns:
        Dictionary berisi semua komponen UI yang telah dibuat
    """
    current_config = config or {}
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="🚀 Environment Setup",
        subtitle="Konfigurasi lingkungan untuk SmartCash YOLOv5-EfficientNet",
        status_message="Siap mengatur environment",
        status_type="info"
    )
    
    # 2. Create Form Container for main content
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="0",
        gap="10px"
    )
    
    # Add panels to form container
    env_info_panel = create_env_info_panel()
    tips_panel = create_tips_requirements()
    setup_summary = create_setup_summary()
    
    form_container['add_item'](env_info_panel, height="auto")
    form_container['add_item'](tips_panel, height="auto")
    form_container['add_item'](setup_summary, height="auto")
    
    # 3. Create Action Container
    action_container = create_action_container(
        buttons=[
            {
                'button_id': 'setup',
                'text': '🚀 Setup Environment',
                'style': 'primary',
                'order': 1,
                'phases': [
                    {
                        'id': 'start',
                        'text': 'Start Setup',
                        'icon': '🚀',
                        'variant': 'primary',
                        'tooltip': 'Mulai proses setup environment',
                        'disabled': False
                    },
                    {
                        'id': 'in_progress',
                        'text': 'Memproses...',
                        'icon': '⏳',
                        'variant': 'primary',
                        'disabled': True,
                        'loading': True
                    },
                    {
                        'id': 'completed',
                        'text': 'Setup Selesai',
                        'icon': '✅',
                        'variant': 'success',
                        'disabled': False,
                        'tooltip': 'Setup environment berhasil'
                    },
                    {
                        'id': 'failed',
                        'text': 'Setup Gagal',
                        'icon': '❌',
                        'variant': 'danger',
                        'disabled': False,
                        'tooltip': 'Klik untuk coba lagi'
                    }
                ]
            }
        ],
        title="🔧 Environment Actions",
        alignment="center"
    )
    
    # 4. Create Operation Container
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True
    )
    
    # 5. Create Footer Container
    footer_container = create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="ℹ️ Informasi",
                content="""
                <div style="padding: 10px;">
                    <p>Pastikan Anda sudah terhubung ke runtime Colab sebelum memulai setup.</p>
                    <p>Klik tombol <strong>Setup Environment</strong> untuk memulai proses instalasi.</p>
                </div>
                """,
                style="info",
                flex="1",
                min_width="300px",
                open_by_default=True
            )
        ],
        style={"border_top": "1px solid #e0e0e0", "padding": "10px 0"},
        flex_flow="row wrap",
        justify_content="space-between"
    )
    
    # 6. Create Main Container with flexible ordering
    components = [
        # Header at the top
        {
            'type': 'header',
            'component': header_container.container,
            'order': 0
        },
        # Action buttons (moved up)
        {
            'type': 'action',
            'component': action_container['container'],
            'order': 1
        },
        # Form container with all panels (moved down)
        {
            'type': 'form',
            'component': form_container['container'],
            'order': 2
        },
        # Operation container (progress, dialogs, logs)
        {
            'type': 'operation',
            'component': operation_container['container'],
            'order': 3
        },
        # Footer at the bottom
        {
            'type': 'footer',
            'component': footer_container.container,
            'order': 4
        }
    ]
    
    main_container = create_main_container(components=components)
    
    # Create a dictionary with all UI components
    ui_components = {
        # Container components
        'main_container': main_container.container,
        'ui': main_container.container,  # Alias for compatibility
        'header_container': header_container,
        'form_container': form_container,  # Keep for backward compatibility
        'action_container': action_container,
        'footer_container': footer_container,
        'operation_container': operation_container,
        
        # Individual UI elements
        'env_info_panel': env_info_panel,
        'tips_panel': tips_panel,
        'setup_summary': setup_summary,
        
        # Operation components
        'progress_tracker': operation_container['progress_tracker'],
        'progress_bar': operation_container['progress_tracker'].progress_bar,  # For SetupHandler
        'status_label': operation_container['progress_tracker'].status_label,  # For SetupHandler
        'confirmation_dialog': operation_container['dialog'],
        'log_accordion': operation_container['log_accordion'],
        'summary_widget': setup_summary  # For SetupHandler
    }
    
    # Add the setup button from action container if it exists
    if 'setup_button' in action_container:
        ui_components['setup_button'] = action_container['setup_button']
    
    return ui_components
