"""
file_path: smartcash/ui/setup/colab/components/colab_ui.py
Description: UI components for Colab environment configuration

This module provides a standardized UI for configuring the Google Colab environment
for SmartCash. It follows the SmartCash UI component architecture and design patterns.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Standard UI components
import ipywidgets as widgets
from IPython.display import display

# SmartCash UI components
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container

# Error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.errors.enums import ErrorLevel

# Local imports
from smartcash.ui.setup.colab.constants import UI_CONFIG, BUTTON_CONFIG
from smartcash.ui.setup.colab.components.setup_summary import create_setup_summary
from smartcash.ui.setup.colab.components.env_info_panel import create_env_info_panel
from smartcash.ui.setup.colab.components.tips_panel import create_tips_requirements

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG

def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets for the Colab setup module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing form widgets and rows
    """
    # Create form widgets
    env_info_panel = create_env_info_panel()
    tips_panel = create_tips_requirements()
    setup_summary = create_setup_summary()
    
    # Create form rows
    form_rows = [
        [env_info_panel],
        [setup_summary],
        # [tips_panel]  # Uncomment if needed
    ]
    
    return {
        'components': {
            'env_info_panel': env_info_panel,
            'tips_panel': tips_panel,
            'setup_summary': setup_summary
        },
        'form_rows': form_rows
    }


def _create_module_summary_content(components: Dict[str, Any]) -> str:
    """Create summary content for the module.
    
    Args:
        components: Dictionary of UI components
        
    Returns:
        HTML string containing the summary content
    """
    return "<p>Environment setup configuration and status will be displayed here.</p>"


def _create_module_info_box() -> widgets.Widget:
    """Create an info box with module documentation.
    
    Returns:
        Info box widget
    """
    return widgets.HTML(
        value="""
        <div style="padding: 12px; background: #e3f2fd; border-radius: 4px; margin: 8px 0;">
            <h4 style="margin-top: 0; color: #0d47a1;">Environment Setup Guide</h4>
            <p>This module helps you set up your Google Colab environment for SmartCash.</p>
            <ol style="margin: 8px 0 0 16px; padding-left: 8px;">
                <li>Configure your environment settings</li>
                <li>Click 'Setup Environment' to initialize</li>
                <li>Follow the progress in the logs</li>
            </ol>
        </div>
        """
    )


@handle_ui_errors(
    error_component_title=f"{UI_CONFIG['module_name']} Error",
    log_error=True,
    return_type=dict,
    level=ErrorLevel.ERROR,
    fail_fast=False,
    create_ui=True
)
def create_colab_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create the Colab environment setup UI.
    
    This function creates a complete UI for configuring the Colab environment
    with the following sections:
    - Environment information
    - Setup summary
    - Action buttons
    
    Args:
        config: Optional configuration dictionary with initial values
        **kwargs: Additional keyword arguments passed to container components
        
    Returns:
        Dictionary containing all UI components and containers
    """
    # Initialize config if not provided
    config = config or {}
    
    # Create UI components dictionary with error handling
    ui_components = {}
    
    try:
        # 1. Create form widgets
        form_widgets = _create_module_form_widgets(config)
        components = form_widgets['components']
        form_rows = form_widgets['form_rows']
        
        # 2. Create header container
        header_container = create_header_container(
            title=f"{UI_CONFIG['icon']} {UI_CONFIG['module_name']}",
            subtitle=UI_CONFIG['module_description'],
            status_message="Ready to configure environment",
            status_type="info"
        )
        
        # 3. Create form container
        form_container = create_form_container(
            form_rows=form_rows,
            layout_type=LayoutType.COLUMN,
            container_margin="0",
            container_padding="16px",
            gap="12px"
        )
        
        # 4. Create action container with buttons
        action_buttons = [
            {
                'name': 'setup_button',
                'label': BUTTON_CONFIG['setup']['text'],
                'button_style': BUTTON_CONFIG['setup']['style'],
                'tooltip': BUTTON_CONFIG['setup']['tooltip'],
                'icon': 'rocket'
            },
            {
                'name': 'save_button',
                'label': BUTTON_CONFIG['save']['text'],
                'button_style': BUTTON_CONFIG['save']['style'],
                'tooltip': BUTTON_CONFIG['save']['tooltip'],
                'icon': 'save'
            },
            {
                'name': 'reset_button',
                'label': BUTTON_CONFIG['reset']['text'],
                'button_style': BUTTON_CONFIG['reset']['style'],
                'tooltip': BUTTON_CONFIG['reset']['tooltip'],
                'icon': 'undo'
            }
        ]
        
        action_container = create_action_container(
            buttons=action_buttons,
            title="Environment Actions",
            alignment="center"
        )
        
        # 5. Create operation container for logs and progress
        operation_container = create_operation_container(
            title="Setup Progress",
            show_logs=True,
            show_progress=True,
            collapsible=True,
            collapsed=False
        )
        
        # 6. Create summary container
        summary_content = _create_module_summary_content(components)
        summary_container = create_summary_container(
            title="Environment Overview",
            theme="primary",
            icon="📊"
        )
        summary_container.set_content(summary_content)
        
        # 7. Create footer with info box
        info_box = _create_module_info_box()
        footer_container = create_footer_container(
            info_box=info_box,
            show_tips=True,
            show_version=True
        )
        
        # 8. Create main container and assemble all components
        main_container = create_main_container(
            header=header_container.container,
            body=widgets.VBox([
                form_container['container'],
                action_container['container'],
                summary_container.container,
                operation_container['container']
            ]),
            footer=footer_container,
            container_config={
                'margin': '0 auto',
                'max_width': '1200px',
                'padding': '10px',
                'border': '1px solid #e0e0e0',
                'border_radius': '5px',
                'box_shadow': '0 1px 3px rgba(0,0,0,0.1)'
            }
        )
        
        # Get buttons from action container
        buttons = {}
        if hasattr(action_container, 'get'):
            for btn in action_buttons:
                buttons[btn['name']] = action_container.get(btn['name'])
        
        # Prepare the UI components dictionary
        ui_components.update({
            # Core containers
            'ui': main_container,
            'main_container': main_container,
            'header_container': header_container.container,
            'form_container': form_container['container'],
            'action_container': action_container['container'],
            'operation_container': operation_container['container'],
            'summary_container': summary_container.container,
            'footer_container': footer_container,
            
            # Form components
            'form_components': components,
            'form_rows': form_rows,
            
            # Buttons
            'buttons': buttons,
            'setup_button': buttons.get('setup_button'),
            'save_button': buttons.get('save_button'),
            'reset_button': buttons.get('reset_button'),
            
            # Additional UI elements
            'summary_content': summary_content,
            'info_box': info_box,
            'log_accordion': operation_container.get('log_accordion'),
            
            # Module metadata
            'module_name': UI_CONFIG['module_name'],
            'parent_module': UI_CONFIG['parent_module'],
            'version': UI_CONFIG['version'],
            'ui_initialized': True
        })
        
        return ui_components
    
    except Exception as e:
        # The error will be handled by the @handle_ui_errors decorator
        raise


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

