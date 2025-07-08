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
    Create Colab UI components with new container
    
    Args:
        config: Configuration for UI components
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing all created UI components
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
    form_container['add_item'](tips_panel, height="auto")
    form_container['add_item'](setup_summary, height="auto")
    
    # 3. Create Action Container
    action_container = create_action_container(
        buttons=[],  # No additional buttons, use the primary button with phases
        title="🚀 Environment Setup",
        alignment="center",
        phases=[
            'init', 'drive', 'symlink', 'folders', 
            'config', 'env', 'verify', 'complete', 'error'
        ]
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
    footer_container = create_footer_container(
        left_text="SmartCash Environment Setup v1.0",
        right_text=f" 2023-{datetime.now().year} SmartCash Team"
    )
    
    # 6. Create Main Container and assemble all components
    main_container = create_main_container(
        header=header_container['container'],
        content=widgets.VBox([
            form_container['container'],
            action_container['container'],
            operation_container['container']
        ]),
        footer=footer_container['container'],
        container_config={
            'margin': '0 auto',
            'max_width': '1200px',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'box_shadow': '0 1px 3px rgba(0,0,0,0.1)'
        }
    )
    
    # Prepare the components dictionary
    ui_components = {
        'main_container': main_container,
        'header_container': header_container['container'],
        'form_container': form_container['container'],
        'action_container': action_container['container'],
        'setup_button': action_container.get('primary_button'),
        'action_container_manager': action_container,
        'operation_container': operation_container['container'],
        'footer_container': footer_container['container'],
        'env_info_panel': env_info_panel,
        'tips_panel': tips_panel,
        'setup_summary': setup_summary,
        'ui': main_container  # Main UI reference
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

def create_colab_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create colab UI components following dependency pattern with operation_container integration.
    
    Args:
        config: Configuration for UI components
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing all created UI components
    """
    current_config = config or {}
    child_components = {}
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="🚀 Environment Setup",
        subtitle="Configure environment for SmartCash YOLOv5-EfficientNet",
        status_message="Ready to setup environment",
        status_type="info"
    )
    child_components['header_container'] = header_container.container
    
    # 2. Create Operation Container (centralized progress and logging)
    operation_container = create_operation_container(
        component_name="colab_operation_container",
        show_progress=True,
        show_logs=True,
        initial_message="Environment setup ready to begin...",
        log_height="200px"
    )
    child_components['operation_container'] = operation_container['container']
    child_components['operation_manager'] = operation_container  # For handlers to access
    
    # 3. Create Environment Form Components
    environment_form = create_environment_form(current_config)
    child_components.update(environment_form)
    
    # 4. Create Setup Form Components  
    setup_form = create_setup_form(current_config)
    child_components.update(setup_form)
    
    # 5. Create two-column layout for forms
    two_column_layout = widgets.HBox([
        widgets.Box(
            [environment_form['environment_form']],
            layout=widgets.Layout(width='50%', padding='0 5px 0 0')
        ),
        widgets.Box(
            [setup_form['setup_form']],
            layout=widgets.Layout(width='50%', padding='0 0 0 5px')
        )
    ], layout=widgets.Layout(
        display='flex',
        gap='10px',
        width='100%',
        align_items='flex-start'
    ))
    
    # 6. Create Form Container
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="0",
        gap="10px"
    )
    form_container['form_container'].children = (two_column_layout,)
    child_components['form_container'] = form_container['container']
    
    # 7. Create status summary
    status_summary = create_setup_summary()
    child_components['status_summary'] = status_summary
    
    # 8. Create save/reset buttons
    save_reset_components = create_save_reset_buttons(
        save_label="💾 Save Config",
        reset_label="🔄 Reset", 
        with_sync_info=True
    )
    child_components['save_reset_buttons'] = save_reset_components
    
    config_buttons_container = widgets.Box(
        [save_reset_components['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%', margin='8px 0')
    )
    child_components['config_buttons_container'] = config_buttons_container
    child_components['save_button'] = save_reset_components.get('save_button')
    child_components['reset_button'] = save_reset_components.get('reset_button')
    
    # 9. Create Action Container with single main action button using phases
    action_container = create_action_container(
        buttons=[],  # No additional buttons, use the primary button with phases
        title="🚀 Environment Setup",
        alignment="center",
        phases=[
            'init', 'drive', 'symlink', 'folders', 
            'config', 'env', 'verify', 'complete', 'error'
        ]
    )
    child_components['action_container'] = action_container['container']
    child_components['setup_button'] = action_container.get('primary_button')
    child_components['action_container_manager'] = action_container  # For phase management
    
    # 10. Create Environment Info Panel
    env_info_panel = create_env_info_panel()
    child_components['env_info_panel'] = env_info_panel
    
    # 11. Create Tips Panel
    tips_panel = create_tips_requirements()
    child_components['tips_panel'] = tips_panel
    
    # 12. Create Footer Container
    footer_container = create_footer_container(
        info_box=widgets.HTML(
            value="""
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>💡 Environment Setup Tips:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>Single click runs: INIT → DRIVE → SYMLINK → FOLDERS → CONFIG → ENV → VERIFY → COMPLETE</li>
                    <li>Progress tracking and detailed logs are shown in the operation container</li>
                    <li>Setup will automatically verify all components at the end</li>
                    <li>Any issues will be reported with specific remediation steps</li>
                </ul>
            </div>
            """
        )
    )
    child_components['footer_container'] = footer_container.container
    
    # 13. Create Main Container using main_container component
    main_container = create_main_container(
        # Define the layout structure with explicit ordering
        components=[
            # Header section
            {
                'type': 'header',
                'component': child_components['header_container'],
                'order': 0,
                'name': 'header'
            },

            # Form section
            {
                'type': 'form',
                'component': child_components['form_container'],
                'order': 2,
                'name': 'form'
            },
            # Info and summary section (custom layout)
            {
                'type': 'custom',
                'component': widgets.HBox([
                    widgets.VBox([
                        child_components['env_info_panel'],
                        child_components['tips_panel']
                    ], layout=widgets.Layout(width='50%', padding='0 10px 0 0')),
                    widgets.VBox([
                        child_components['status_summary']
                    ], layout=widgets.Layout(width='50%', padding='0 0 0 10px'))
                ], layout=widgets.Layout(width='100%', margin='10px 0')),
                'order': 3,
                'name': 'info_summary'
            },
            # Config buttons
            {
                'type': 'action',
                'component': child_components['config_buttons_container'],
                'order': 4,
                'name': 'config_buttons'
            },
            # Main action container
            {
                'type': 'action',
                'component': child_components['action_container'],
                'order': 5,
                'name': 'actions'
            },
            # Operation container (progress and logs)
            {
                'type': 'operation',
                'component': child_components['operation_container'],
                'order': 6,
                'name': 'operations'
            },
            # Footer
            {
                'type': 'footer',
                'component': child_components['footer_container'],
                'order': 7,
                'name': 'footer'
            }
        ],
        # Styling options
        container_style={
            'width': '100%',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border_radius': '10px'
        }
    )
    
    # Store the main container and its UI reference
    child_components['main_container'] = main_container.container
    child_components['ui'] = main_container.container  # Main UI reference
    child_components['main_container_manager'] = main_container  # For programmatic control
    
    return child_components
