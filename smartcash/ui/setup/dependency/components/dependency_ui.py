"""
File: smartcash/ui/setup/dependency/components/dependency_ui.py
Description: Dependency management UI following SmartCash standardized template.

This module provides the user interface for managing Python package dependencies
in the SmartCash environment, including installation, updates, and verification.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Dependency Tabs)
3. Action Container (Install/Check/Uninstall Buttons)
4. Operation Container (Progress + Logs)
5. Footer Container (Tips and Info)
"""

from typing import Optional, Dict, Any, Tuple, List
import ipywidgets as widgets

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container, ActionContainer
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Module-specific imports
from .dependency_tabs import create_dependency_tabs
from .operation_summary import create_operation_summary

# === MODULE CONSTANTS ===
UI_CONFIG = {
    'title': "Dependency Manager",
    'subtitle': "Install, update, and manage Python packages",
    'icon': "📦",
    'module_name': "dependency",
    'parent_module': "setup",
    'version': "1.0.0"
}

# Button configuration - Using multiple action buttons pattern
BUTTON_CONFIG = {
    'install': {
        'text': 'Install',
        'style': 'success',
        'icon': 'download',
        'tooltip': 'Install selected packages',
        'order': 1
    },
    'check_updates': {
        'text': 'Check & Updates',
        'style': 'info',
        'icon': 'refresh',
        'tooltip': 'Check package status and available updates',
        'order': 2
    },
    'uninstall': {
        'text': 'Uninstall',
        'style': 'danger',
        'icon': 'trash',
        'tooltip': 'Uninstall selected packages',
        'order': 3
    }
}

@handle_ui_errors(error_component_title="Dependency UI Error")
def create_dependency_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the dependency management UI following SmartCash standards.
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references with 'ui_components' key
        
    Example:
        >>> ui = create_dependency_ui()
        >>> display(ui['ui'])  # Display the UI
    """
    # Initialize configuration and components dictionary
    current_config = config or {}
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Create Header Container ===
    header_container = create_header_container(
        title=f"{UI_CONFIG['icon']} {UI_CONFIG['title']}",
        subtitle=UI_CONFIG['subtitle'],
        status_message="Ready to manage dependencies",
        status_type="info"
    )
    # Store both the container object and its widget
    ui_components['containers']['header'] = {
        'container': header_container.container,
        'widget': header_container
    }
    
    # === 2. Create Form Container ===
    # Create form widgets
    form_widgets = _create_module_form_widgets(current_config)
    
    # Create form container with the widgets
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    form_container['container'].children = (form_widgets['ui'],)
    
    # Store references
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widgets'])
    
    # === 3. Create Action Container ===
    # Create action container with buttons
    action_container = ActionContainer(
        container_margin="12px 0",
        show_save_reset=True
    )
    
    # Add buttons from BUTTON_CONFIG
    for button_id, btn_config in BUTTON_CONFIG.items():
        action_container.add_button(
            button_id,  # Pass as positional argument
            text=btn_config['text'],
            style=btn_config['style'],
            icon=btn_config['icon'],
            tooltip=btn_config['tooltip'],
            order=btn_config['order'],
            disabled=False
        )
    
    # Format the container as expected by the rest of the code
    action_container = {
        'container': action_container.container,
        'buttons': action_container.buttons,
        'primary_button': action_container.buttons.get('primary'),
        'action_container': action_container,
        'set_phase': action_container.set_phase,
        'set_phases': action_container.set_phases,
        'enable_all': action_container.enable_all,
        'disable_all': action_container.disable_all,
        'set_all_buttons_enabled': action_container.set_all_buttons_enabled
    }
    
    # Store button references
    ui_components['containers']['actions'] = action_container
    for button_id, button in action_container['buttons'].items():
        ui_components['widgets'][f'{button_id}_button'] = button
    
    # === 4. Create Operation Container ===
    operation_container = create_operation_container(
        component_name="dependency_operation_container",
        show_progress=True,
        show_logs=True,
        initial_message="Dependency manager ready...",
        log_height="200px"
    )
    ui_components['containers']['operation'] = operation_container
    
    # === 5. Create Summary Container ===
    summary_content = _create_module_summary_content(current_config)
    ui_components['containers']['summary'] = summary_content
    
    # === 6. Create Footer Container ===
    footer_container = create_footer_container(
        info_box=_create_module_info_box(),
        tips_box=_create_module_tips_box()
    )
    # Store both the container object and its widget
    ui_components['containers']['footer'] = {
        'container': footer_container.container,
        'widget': footer_container
    }
    
    # === 7. Create Main Container ===
    main_container = create_main_container(
        components=[
            # Header
            {
                'type': 'header',
                'component': ui_components['containers']['header']['container'],
                'order': 0,
                'name': 'header'
            },
            # Form
            {
                'type': 'form',
                'component': ui_components['containers']['form']['container'],
                'order': 1,
                'name': 'form'
            },
            # Action Buttons
            {
                'type': 'action',
                'component': ui_components['containers']['actions']['container'],
                'order': 2,
                'name': 'actions'
            },
            # Summary
            {
                'type': 'summary',
                'component': ui_components['containers']['summary'],
                'order': 3,
                'name': 'summary'
            },
            # Operation (progress + logs)
            {
                'type': 'operation',
                'component': ui_components['containers']['operation']['container'],
                'order': 4,
                'name': 'operations'
            },
            # Footer
            {
                'type': 'footer',
                'component': ui_components['containers']['footer']['container'],
                'order': 5,
                'name': 'footer'
            }
        ],
        container_style={
            'width': '100%',
            'padding': '20px',
            'border': '1px solid #e0e0e0',
            'border_radius': '8px',
            'box_shadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    )
    
    # Store main UI references
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container
    
    result = {
        'ui_components': ui_components,
        'ui': ui_components['ui']
    }
    
    # Add all components to the root for backward compatibility
    result.update(ui_components['containers'])
    result.update(ui_components['widgets'])
    
    return result


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    # Create dependency tabs (module-specific form content)
    dependency_tabs = create_dependency_tabs(config)
    
    # Create a container for the form
    form_container = widgets.VBox([dependency_tabs])
    
    return {
        'ui': form_container,
        'widgets': {
            'dependency_tabs': dependency_tabs
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> widgets.Widget:
    """
    Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Widget containing the summary content
    """
    return create_operation_summary("Dependency operations ready...")


def _create_module_info_box() -> widgets.Widget:
    """
    Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    return widgets.HTML(
        value="""
        <div style="font-size: 0.9em; line-height: 1.5;">
            <p><strong>Dependency Manager v1.0.0</strong></p>
            <p>Manage Python packages and their dependencies for SmartCash.</p>
            <p>Modules: Core, ML, Data, Visualization</p>
        </div>
        """
    )


def _create_module_tips_box() -> widgets.Widget:
    """
    Create the tips box content for the footer.
    
    Returns:
        Widget containing the tips content
    """
    return widgets.HTML(
        value="""
        <div style="font-size: 0.9em; line-height: 1.5;">
            <p><strong>💡 Tips for Managing Dependencies:</strong></p>
            <ul style="margin: 5px 0 0 15px; padding: 0;">
                <li>Use the Categories tab for recommended package collections</li>
                <li>Check for updates regularly to keep packages secure</li>
                <li>Review dependency conflicts in the log before installing</li>
                <li>Default packages (⭐) are recommended for best compatibility</li>
            </ul>
        </div>
        """
    )
