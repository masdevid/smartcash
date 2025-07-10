"""
File: smartcash/ui/setup/colab/components/colab_ui.py
Description: Consolidated UI components for Colab environment configuration
"""

from __future__ import annotations

import ipywidgets as widgets
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import container components
from smartcash.ui.components.main_container import (
    create_main_container,
    ContainerConfig
)
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

# Import local colab components
from smartcash.ui.setup.colab.components.setup_summary import create_setup_summary
from smartcash.ui.setup.colab.components.env_info_panel import create_env_info_panel
from smartcash.ui.setup.colab.components.tips_panel import create_tips_requirements

def create_colab_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create Colab UI components following standard container order.
    
    Standard Order (per docs/ui/planning/ui_module_structure.md):
    1. Header Container (Header + Status Panel) - Required
    2. Form Container (Custom to module) - Required  
    3. Action Container (Save/Reset | Primary | Action Buttons) - Required
    4. Summary Container (Custom, Nice to have) - Optional
    5. Operation Container (Progress + Dialog + Log) - Required
    6. Footer Container (Info Accordion + Tips) - Optional
    
    Args:
        config: Configuration for UI components
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing all created UI components in standard structure
    """
    current_config = config or {}
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="🚀 Environment Setup",
        subtitle="Configure environment for SmartCash YOLOv5-EfficientNet",
        status_message="Ready to setup environment",
        status_type="info"
    )
    
    # 2. Create Form Container for main content
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    
    # Add panels to form container
    env_info_panel = create_env_info_panel()
    tips_panel = create_tips_requirements()
    setup_summary = create_setup_summary()
    
    form_container['add_item'](env_info_panel, height="auto")
    # form_container['add_item'](tips_panel, height="auto")
    form_container['add_item'](setup_summary, height="auto")
    
    # 3. Create Action Container - use default phases (will be set to COLAB_PHASES automatically)
    action_container = create_action_container(
        buttons=[],  # No additional buttons, use the primary button with phases
        title="🚀 Environment Setup",
        alignment="center"
        # phases parameter removed - will use default COLAB_PHASES
    )
    
    # 4. Create Operation Container for progress and logs
    operation_container = create_operation_container(
        title="Setup Progress",
        show_logs=True,
        show_progress=True,
        collapsible=True,
        collapsed=False
    )
    
    # 5. Create Footer Container
    footer_container = widgets.HTML(
        value=f"""
        <div style="display: flex; justify-content: space-between; align-items: center; 
                    padding: 10px; border-top: 1px solid #e0e0e0; margin-top: 20px; 
                    background-color: #f9f9f9; font-size: 0.9em; color: #666;">
            <span>SmartCash Environment Setup v1.0</span>
            <span>2023-{datetime.now().year} SmartCash Team</span>
        </div>
        """
    )
    
    # 5. Create Summary Container (following standard order)
    summary_container = create_summary_container(
        title="Environment Overview",
        theme="primary",
        icon="📊"
    )
    # Set setup summary content - for now use HTML string, will integrate widget later
    summary_container.set_content("Environment setup configuration and status will be displayed here.")
    
    # 6. Create Main Container and assemble all components (STANDARD ORDER)
    main_container = create_main_container(
        header=header_container.container,
        content=widgets.VBox([
            form_container['container'],        # 2. Form Container
            action_container['container'],      # 3. Action Container  
            summary_container.container,        # 4. Summary Container
            operation_container['container']    # 5. Operation Container
        ]),
        footer=footer_container,               # 6. Footer Container
        container_config={
            'margin': '0 auto',
            'max_width': '1200px',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'box_shadow': '0 1px 3px rgba(0,0,0,0.1)'
        }
    )
    
    # Prepare the components dictionary (STANDARD STRUCTURE)
    ui_components = {
        # Core containers (standard order)
        'ui': main_container,                           # Main UI reference
        'main_container': main_container,               # 1. Main container
        'header_container': header_container.container, # 2. Header container
        'form_container': form_container['container'],  # 3. Form container
        'action_container': action_container['container'], # 4. Action container
        'summary_container': summary_container.container, # 5. Summary container
        'operation_container': operation_container['container'], # 6. Operation container
        'footer_container': footer_container,           # 7. Footer container
        
        # Action components
        'setup_button': action_container.get('primary_button'),
        'action_container_manager': action_container,
        
        # Custom components
        'env_info_panel': env_info_panel,
        'tips_panel': tips_panel,
        'setup_summary': setup_summary,
        
        # Module metadata
        'module_name': 'colab',
        'parent_module': 'setup',
        'ui_initialized': True
    }
    
    return ui_components


def create_environment_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create environment configuration form components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing environment form components
    """
    env_config = config.get('environment', {})
    components = {}
    
    # Environment type dropdown
    components['environment_type_dropdown'] = widgets.Dropdown(
        options=[
            ('Google Colab', 'colab'),
            ('Kaggle Notebooks', 'kaggle'),
            ('Local Environment', 'local')
        ],
        value=env_config.get('type', 'colab'),
        description='Environment:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Project name text input
    components['project_name_text'] = widgets.Text(
        value=env_config.get('project_name', 'SmartCash'),
        description='Project Name:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Auto mount drive checkbox
    components['auto_mount_drive_checkbox'] = widgets.Checkbox(
        value=env_config.get('auto_mount_drive', True),
        description='Auto Mount Drive',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # GPU enabled checkbox
    components['gpu_enabled_checkbox'] = widgets.Checkbox(
        value=env_config.get('gpu_enabled', False),
        description='Enable GPU',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # GPU type dropdown
    components['gpu_type_dropdown'] = widgets.Dropdown(
        options=[
            ('No GPU', 'none'),
            ('Tesla K80', 'k80'),
            ('Tesla T4', 't4'),
            ('Tesla P100', 'p100'),
            ('Tesla V100', 'v100')
        ],
        value=env_config.get('gpu_type', 'none'),
        description='GPU Type:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Create environment form layout
    components['environment_form'] = widgets.VBox([
        widgets.HTML("<h4>🌍 Environment Configuration</h4>"),
        components['environment_type_dropdown'],
        components['project_name_text'],
        components['auto_mount_drive_checkbox'],
        components['gpu_enabled_checkbox'],
        components['gpu_type_dropdown']
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', border_radius='5px'))
    
    return components


def create_setup_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create setup configuration form components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing setup form components
    """
    setup_config = config.get('setup', {})
    components = {}
    
    # Setup stages selection
    components['setup_stages_select'] = widgets.SelectMultiple(
        options=[
            ('Environment Detection', 'environment_detection'),
            ('Drive Mount', 'drive_mount'),
            ('GPU Setup', 'gpu_setup'),
            ('Folder Setup', 'folder_setup'),
            ('Config Sync', 'config_sync'),
            ('Verify', 'verify')
        ],
        value=tuple(setup_config.get('stages', [])),
        description='Setup Stages:',
        rows=6,
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Auto start checkbox
    components['auto_start_checkbox'] = widgets.Checkbox(
        value=setup_config.get('auto_start', False),
        description='Auto Start Setup',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Stop on error checkbox
    components['stop_on_error_checkbox'] = widgets.Checkbox(
        value=setup_config.get('stop_on_error', True),
        description='Stop on Error',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Max retries input
    components['max_retries_int'] = widgets.IntText(
        value=setup_config.get('max_retries', 3),
        description='Max Retries:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Show advanced options checkbox
    ui_config = config.get('ui', {})
    components['show_advanced_checkbox'] = widgets.Checkbox(
        value=ui_config.get('show_advanced_options', False),
        description='Show Advanced Options',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Create setup form layout
    components['setup_form'] = widgets.VBox([
        widgets.HTML("<h4>⚙️ Setup Configuration</h4>"),
        components['setup_stages_select'],
        components['auto_start_checkbox'],
        components['stop_on_error_checkbox'],
        components['max_retries_int'],
        components['show_advanced_checkbox']
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', border_radius='5px'))
    
    return components

