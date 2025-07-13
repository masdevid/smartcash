"""
File: smartcash/ui/setup/colab/components/colab_ui.py
Description: Main UI assembly for Colab environment setup

This module serves as the main entry point for the Colab UI, assembling all
components into a complete interface. It follows the standardized container-based
architecture with consistent ordering and structure.

Components are split into separate modules for better maintainability:
- colab_ui_environment.py: Environment information container
- colab_ui_form.py: Form widgets and configuration
- colab_ui_summary.py: Summary content
- colab_ui_footer.py: Footer components
- colab_ui_helpers.py: Helper functions and validators
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container

# Local component imports
from .colab_ui_environment import create_environment_container as _create_environment_container
from .colab_ui_form import create_module_form_widgets as _create_module_form_widgets
from .colab_ui_summary import create_module_summary_content as _create_module_summary_content
from .colab_ui_footer import (
    create_module_info_box as _create_module_info_box,
    create_module_tips_box as _create_module_tips_box
)
from .colab_ui_helpers import (
    get_colab_default_config,
    validate_colab_config,
    update_colab_config
)

# Error handling import
from smartcash.ui.core.errors.handlers import handle_ui_errors

# UI Configuration
UI_CONFIG = {
    'title': 'Google Colab Setup',
    'description': 'Configure your Google Colab environment for SmartCash',
    'module_name': 'colab',
    'parent_module': 'setup',
    'version': '1.0.0',
    'icon': '⚙️'  # Font Awesome icon name
}

# Button configuration
BUTTON_CONFIG = {
    'primary': {
        'description': 'Initialize Environment',
        'button_style': 'success',
        'icon': 'check'
    },
    'secondary': {
        'description': 'Reset',
        'button_style': 'warning',
        'icon': 'undo'
    },
    'cancel': {
        'description': 'Cancel',
        'button_style': 'danger',
        'icon': 'times'
    }
}


# Re-export the functions for backward compatibility
create_environment_container = _create_environment_container
create_module_form_widgets = _create_module_form_widgets
create_module_summary_content = _create_module_summary_content
create_module_info_box = _create_module_info_box
create_module_tips_box = _create_module_tips_box


@handle_ui_errors(error_component_title="Colab UI Creation Error")
def create_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create Colab UI using standardized container architecture.
    
    This function creates the complete UI for the Colab module following the
    standardized container order and structure. It provides consistent layout,
    styling, and functionality across all SmartCash UI modules.
    
    Container Order (standardized):
    1. Header Container (Header + Status Panel)
    2. Form Container (Custom to each module)
    3. Action Container (Save/Reset | Primary | Action Buttons)
    4. Summary Container (Custom, Nice to have)
    5. Operation Container (Progress + Dialog + Log)
    6. Footer Container (Info Accordion + Tips)
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments for customization
        
    Returns:
        Dictionary containing all UI components and widgets with standardized keys:
        - 'ui': Main UI widget
        - 'header_container': Header container widget
        - 'form_container': Form container widget
        - 'action_container': Action container widget
        - 'summary_container': Summary container widget (if enabled)
        - 'operation_container': Operation container widget
        - 'footer_container': Footer container widget
        - Individual widget references for handlers
        - Module metadata and configuration
    """
    config = config or {}
    ui_components = {}
    
    # === 1. HEADER CONTAINER ===
    # Header with title, subtitle, and status indicator
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['description'],
        icon=UI_CONFIG['icon'],
        status_message="Ready for environment setup",
        status_type="info",
        show_status_panel=True
    )
    # Store both the container widget and the header_container object for status updates
    ui_components['header_container'] = header_container
    ui_components['main_header_widget'] = header_container.container
    
    # === 2. FORM CONTAINER ===
    # Custom form layout specific to the module
    form_container = create_form_container(
        title=f"⚙️ {UI_CONFIG['title']} Configuration",
        layout_type="column",  # or "row", "grid" based on needs
        container_padding="15px",
        gap="12px"
    )
    
    # Create module-specific form widgets
    form_widgets = _create_module_form_widgets(config)
    
    # Add form widgets to container
    if form_widgets and 'form_ui' in form_widgets:
        form_container['add_item'](form_widgets['form_ui'], width='100%')
    
    ui_components['form_container'] = form_container['container']
    ui_components['form_widgets'] = form_widgets
    
    # === 3. ACTION CONTAINER ===
    # Primary button for Colab setup with phases - following single operation pattern
    action_container = create_action_container(
        buttons=[{
            'id': 'colab_setup',
            'text': BUTTON_CONFIG['primary']['description'],
            'style': 'primary',
            'tooltip': BUTTON_CONFIG['primary']['description']
        }],
        title=f"🚀 {UI_CONFIG['title']} Operations",
        show_save_reset=True  # Include save/reset for configuration
    )
    ui_components['action_container'] = action_container['container']
    
    # Extract button references
    ui_components['primary_button'] = action_container['primary_button']
    ui_components['setup_button'] = action_container['buttons'].get('colab_setup')
    
    # Save/Reset buttons (access from action_container instance)
    action_container_instance = action_container['action_container']
    ui_components['save_button'] = getattr(action_container_instance, 'save_button', None)
    ui_components['reset_button'] = getattr(action_container_instance, 'reset_button', None)
    
    # Expose phase management methods for primary button
    ui_components['set_phase'] = action_container.get('set_phase')
    ui_components['set_phases'] = action_container.get('set_phases')
    ui_components['enable_all'] = action_container.get('enable_all')
    ui_components['disable_all'] = action_container.get('disable_all')
    
    # === 4. SUMMARY CONTAINER (Optional) ===
    # Custom summary container for displaying module status/results
    summary_enabled = config.get('show_summary', True)
    if summary_enabled:
        summary_container = create_summary_container(
            title=f"{UI_CONFIG['title']} Summary",
            icon="📊"
        )
        
        # Add module-specific summary content
        summary_content = _create_module_summary_content(config)
        if summary_content:
            summary_container.set_content(summary_content.value)
        
        ui_components['summary_container'] = summary_container.container
    
    # === 5. OPERATION CONTAINER ===
    # Progress tracking, dialog, and logging with standardized configuration
    operation_container = create_operation_container(
        show_progress=True,  # Enable progress tracking
        show_dialog=True,    # Enable dialog functionality
        show_logs=True,      # Enable logging
        log_module_name=UI_CONFIG['module_name'],  # Module name for log filtering
        log_namespace_filter='colab',  # Filter logs for colab namespace only
        progress_levels='dual'  # Use dual-level progress tracking
    )
    
    # Store the main container reference
    ui_components['operation_container'] = operation_container['container']
    
    # Store operation container components and functions for easy access
    operation_components = {
        # Core container
        'container': operation_container['container'],
        
        # Progress tracking
        'progress_tracker': operation_container['progress_tracker'],
        'update_progress': operation_container['update_progress'],
        
        # Logging
        'log_accordion': operation_container['log_accordion'],
        'log_message': operation_container['log_message'],
        
        # Dialogs
        'show_dialog': operation_container['show_dialog'],
        'show_info_dialog': operation_container['show_info_dialog'],
        'clear_dialog': operation_container['clear_dialog'],
        'dialog_area': operation_container.get('dialog_area')
    }
    
    # Update ui_components with all operation container components
    ui_components.update(operation_components)
    
    # Log initialization
    ui_components['log_message'](
        f"{UI_CONFIG['module_name']} UI initialized successfully.",
        level='info'
    )
    
    # === 5.1 ENVIRONMENT CONTAINER ===
    # Environment information and status
    environment_container = _create_environment_container(config)
    ui_components['environment_container'] = environment_container
    
    # Add refresh button to update environment info
    refresh_button = widgets.Button(
        description='🔄 Refresh Environment Info',
        button_style='info',
        tooltip='Click to refresh environment information',
        layout=widgets.Layout(width='auto', margin='5px 0')
    )
    
    def on_refresh_clicked(button):
        with environment_container.hold_trait_notifications():
            environment_container.children = [widgets.HTML(
                value='<div style="text-align: center; padding: 10px;">Refreshing environment information...</div>',
                layout={'width': '100%'}
            )]
            new_environment_container = _create_environment_container(config)
            environment_container.children = new_environment_container.children
    
    refresh_button.on_click(on_refresh_clicked)
    
    # Add refresh button to the environment container
    if hasattr(environment_container, 'children'):
        environment_container.children = tuple([refresh_button] + list(environment_container.children))
    
    # === 6. FOOTER CONTAINER ===
    # Info accordion and helpful tips
    from smartcash.ui.components.footer_container import PanelConfig, PanelType
    
    footer_container = create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Environment Setup Info",
                content=_create_module_info_box().value,  # Extract HTML value
                flex="1",
                min_width="300px",
                open_by_default=False
            ),
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Setup Tips", 
                content=_create_module_tips_box().value,  # Extract HTML value
                flex="1",
                min_width="300px",
                open_by_default=False
            )
        ]
    )
    ui_components['footer_container'] = footer_container.container
    
    # === MAIN UI ASSEMBLY ===
    # Assemble all containers in standardized order, filtering out None values
    main_components = []
    
    # Add containers in order, checking each one is not None
    container_order = [
        ui_components.get('main_header_widget'),  # Use the widget, not the object
        ui_components.get('form_container'), 
        ui_components.get('action_container')
    ]
    
    # Add summary container if enabled
    if summary_enabled and 'summary_container' in ui_components:
        container_order.append(ui_components.get('summary_container'))
    
    # Add remaining containers
    container_order.extend([
        ui_components.get('operation_container'),
        ui_components.get('environment_container'),
        ui_components.get('footer_container')
    ])
    
    # Filter out None values
    main_components = [container for container in container_order if container is not None]
    
    # Create main UI container
    main_ui = widgets.VBox(
        main_components,
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#fafafa',
            padding='15px'
        )
    )
    
    ui_components['ui'] = main_ui
    ui_components['main_container'] = main_ui
    
    # === METADATA AND CONFIGURATION ===
    # Add module metadata and configuration
    ui_components.update({
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'logger_namespace': f"smartcash.ui.{UI_CONFIG['parent_module']}.{UI_CONFIG['module_name']}",
        'ui_initialized': True,
        'config': config,
        'version': UI_CONFIG['version']
    })
    
    return ui_components


# === HELPER FUNCTIONS ===

def validate_colab_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Colab configuration against defined rules.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration validation fails
    """
    return validate_colab_config(config)


def get_colab_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the Colab module.
    
    Returns:
        Default configuration dictionary
    """
    return get_colab_default_config()


def update_colab_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Updated configuration dictionary
    """
    return update_colab_config(ui_components)


# === EXPORT FUNCTIONS ===
def create_colab_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Alias for create_colab_ui for backward compatibility.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components
    """
    return create_colab_ui(config, **kwargs)