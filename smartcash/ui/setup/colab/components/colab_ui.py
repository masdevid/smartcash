"""
File: smartcash/ui/setup/colab/components/colab_ui.py
Description: Consolidated UI components for Colab environment configuration
"""

from __future__ import annotations

import ipywidgets as widgets
from typing import Dict, Any, Optional, List

# Import shared UI components
from smartcash.ui.components import (
    create_main_container,
    create_header_container,
    create_action_container,
    create_operation_container,
    create_footer_container,
    create_form_container,
    create_save_reset_buttons
)
from smartcash.ui.components.form_container import LayoutType

# Import local colab components
from smartcash.ui.setup.colab.components.setup_summary import create_setup_summary
from smartcash.ui.setup.colab.components.env_info_panel import create_env_info_panel

# Handle missing tips_panel gracefully
try:
    from smartcash.ui.setup.colab.components.tips_panel import create_tips_requirements
except ImportError:
    def create_tips_requirements():
        return widgets.HTML(
            value="""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9;">
                <h4>💡 Environment Setup Tips</h4>
                <ul>
                    <li>Ensure Google Drive is accessible for Colab environments</li>
                    <li>Check GPU availability for better performance</li>
                    <li>Verify all required dependencies are installed</li>
                </ul>
            </div>
            """,
            layout=widgets.Layout(margin='10px 0')
        )


def create_colab_ui() -> Dict[str, Any]:
    """
    Create the main Colab UI components using main_container for layout management.
    
    Returns:
        Dictionary containing all UI components and the main container
    """
    # Initialize components dictionary
    ui_components: Dict[str, Any] = {}
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="🚀 Environment Setup",
        subtitle="Configure environment for SmartCash YOLOv5-EfficientNet",
        status_message="Ready to setup environment",
        status_type="info"
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Create Form Container for configuration
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    ui_components['form_container'] = form_container['container']
    
    # 3. Create Action Container with single main action button using phases
    action_container = create_action_container(
        buttons=[],  # No additional buttons, use the primary button with phases
        title="🚀 Environment Setup",
        alignment="center",
        phases=[
            'init', 'drive', 'symlink', 'folders', 
            'config', 'env', 'verify', 'complete', 'error'
        ]
    )
    ui_components['action_container'] = action_container['container']
    ui_components['setup_button'] = action_container.get('primary_button')
    ui_components['action_container_manager'] = action_container  # For phase management
    
    # 4. Create Setup Summary and Environment Info Panel
    setup_summary = create_setup_summary()
    env_info_panel = create_env_info_panel()
    
    # Create a two-column layout for summary and info
    info_summary_layout = widgets.HBox([
        widgets.VBox([
            setup_summary
        ], layout=widgets.Layout(width='50%', padding='0 10px 0 0')),
        widgets.VBox([
            env_info_panel
        ], layout=widgets.Layout(width='50%', padding='0 0 0 10px'))
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    ui_components['setup_summary'] = setup_summary
    ui_components['env_info_panel'] = env_info_panel
    ui_components['info_summary_layout'] = info_summary_layout
    
    # 5. Create Operation Container (for progress and logging)
    operation_container = create_operation_container(
        component_name="colab_operation_container",
        show_progress=True,
        show_logs=True,
        initial_message="Environment setup ready to begin...",
        log_height="200px"
    )
    ui_components['operation_container'] = operation_container['container']
    ui_components['operation_manager'] = operation_container
    
    # 6. Create Tips Panel
    tips_panel = create_tips_requirements()
    ui_components['tips_panel'] = tips_panel
    
    # 7. Create Footer Container
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
    ui_components['footer_container'] = footer_container.container
    
    # 8. Create Main Container using main_container component
    main_container = create_main_container(
        # Use the new components list for better control
        components=[
            # Header section
            {
                'type': 'header',
                'component': ui_components['header_container'],
                'order': 0,
                'name': 'header'
            },
            # Form section
            {
                'type': 'form',
                'component': ui_components['form_container'],
                'order': 1,
                'name': 'form'
            },
            # Action buttons
            {
                'type': 'action',
                'component': ui_components['action_container'],
                'order': 2,
                'name': 'actions'
            },
            # Info and summary section (custom component)
            {
                'type': 'custom',
                'component': ui_components['info_summary_layout'],
                'order': 3,
                'name': 'info_summary'
            },
            # Operation container (progress and logs)
            {
                'type': 'operation',
                'component': ui_components['operation_container'],
                'order': 4,
                'name': 'operations'
            },
            # Tips panel
            {
                'type': 'custom',
                'component': ui_components['tips_panel'],
                'order': 5,
                'name': 'tips'
            },
            # Footer
            {
                'type': 'footer',
                'component': ui_components['footer_container'],
                'order': 6,
                'name': 'footer'
            }
        ],
        # Styling options
        container_style={
            'width': '100%',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border_radius': '10px',
            'overflow_y': 'auto',
            'max_height': '90vh'
        }
    )
    
    # Store the main container and its UI reference
    ui_components['main_container'] = main_container.container
    ui_components['ui'] = main_container.container  # Main UI reference
    ui_components['main_container_manager'] = main_container  # For programmatic control
    
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
    try:
        form_container = create_form_container(
            layout_type=LayoutType.COLUMN,
            container_margin="0",
            container_padding="0",
            gap="10px"
        )
        form_container['form_container'].children = (two_column_layout,)
        child_components['form_container'] = form_container['container']
    except Exception:
        # Fallback if form_container has issues
        child_components['form_container'] = widgets.VBox([two_column_layout])
    
    # 7. Create status summary
    status_summary = create_setup_summary()
    child_components['status_summary'] = status_summary
    
    # 8. Create save/reset buttons
    try:
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
    except Exception:
        # Fallback buttons
        save_button = widgets.Button(description="💾 Save Config", button_style='primary')
        reset_button = widgets.Button(description="🔄 Reset", button_style='warning')
        config_buttons_container = widgets.HBox([save_button, reset_button])
        child_components['config_buttons_container'] = config_buttons_container
        child_components['save_button'] = save_button
        child_components['reset_button'] = reset_button
    
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
            # Operation container (progress and logs)
            {
                'type': 'operation',
                'component': child_components['operation_container'],
                'order': 1,
                'name': 'operations'
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
            # Footer
            {
                'type': 'footer',
                'component': child_components['footer_container'],
                'order': 6,
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
