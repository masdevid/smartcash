"""
File: smartcash/ui/setup/dependency/components/dependency_ui.py
Description: Modern dependency manager UI using latest container standards
"""

from typing import Optional, Dict, Any

# Import shared UI components using latest standards
from smartcash.ui.components import (
    create_main_container,
    create_header_container, 
    create_action_container,
    create_operation_container,
    create_footer_container,
    create_form_container
)
from smartcash.ui.components.form_container import LayoutType

from .dependency_tabs import create_dependency_tabs

def create_dependency_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create dependency UI components following latest container standards.
    
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
        title="📦 Dependency Manager",
        subtitle="Manage packages for SmartCash with modern interface",
        status_message="Ready to manage dependencies",
        status_type="info"
    )
    child_components['header_container'] = header_container.container
    
    # 2. Create Operation Container (centralized progress and logging)
    operation_container = create_operation_container(
        component_name="dependency_operation_container",
        show_progress=True,
        show_logs=True,
        initial_message="Dependency manager ready...",
        log_height="200px"
    )
    child_components['operation_container'] = operation_container['container']
    child_components['operation_manager'] = operation_container  # For handlers to access
    
    # 3. Create dependency tabs with improved design
    dependency_tabs = create_dependency_tabs(current_config)
    child_components['dependency_tabs'] = dependency_tabs
    
    # 4. Create Form Container to hold the tabs
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    form_container['container'].children = (dependency_tabs,)
    child_components['form_container'] = form_container['container']
    
    # 5. Create Action Container with multiple operation buttons and default save/reset
    action_container = create_action_container(
        buttons=[
            {
                'id': 'install_button',
                'text': '📥 Install',
                'style': 'success',
                'icon': 'download',
                'tooltip': 'Install selected packages',
                'order': 1
            },
            {
                'id': 'check_button', 
                'text': '🔍 Check & Updates',
                'style': 'info',
                'icon': 'refresh',
                'tooltip': 'Check package status and available updates',
                'order': 2
            },
            {
                'id': 'uninstall_button',
                'text': '🗑️ Uninstall',
                'style': 'danger',
                'icon': 'trash',
                'tooltip': 'Uninstall selected packages',
                'order': 3
            }
        ],
        title="🚀 Package Operations", 
        alignment="center",
        show_save_reset=True  # Use default save/reset buttons
    )
    child_components['action_container'] = action_container['container']
    child_components['install_button'] = action_container['buttons'].get('install_button')
    child_components['check_button'] = action_container['buttons'].get('check_button')
    child_components['uninstall_button'] = action_container['buttons'].get('uninstall_button')
    child_components['action_container_manager'] = action_container  # For button management
    
    # Access save/reset buttons through action container
    child_components['save_button'] = action_container['action_container'].save_button
    child_components['reset_button'] = action_container['action_container'].reset_button
    
    # 7. Create Footer Container with enhanced tips
    import ipywidgets as widgets
    footer_container = create_footer_container(
        info_box=widgets.HTML(
            value="""
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>💡 Dependency Management Tips:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>Use Categories tab for predefined package collections</li>
                    <li>Use Custom tab for individual packages and git repositories</li>
                    <li>Default packages (⭐) are recommended for optimal performance</li>
                    <li>Real-time status tracking shows installation progress</li>
                    <li>Check updates regularly to maintain security and compatibility</li>
                </ul>
            </div>
            """
        )
    )
    child_components['footer_container'] = footer_container.container
    
    # 8. Create Operation Summary for displaying results
    from .operation_summary import create_operation_summary
    operation_summary = create_operation_summary("🔧 Dependency operations ready...")
    child_components['operation_summary'] = operation_summary
    
    # 9. Create Main Container using latest pattern
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
            # Form section (tabs)
            {
                'type': 'form',
                'component': child_components['form_container'],
                'order': 2,
                'name': 'form'
            },
            # Config buttons (if available)
            {
                'type': 'action',
                'component': child_components.get('save_reset_buttons', {}).get('container'),
                'order': 3,
                'name': 'config_buttons'
            },
            # Main action container
            {
                'type': 'action',
                'component': child_components['action_container'],
                'order': 4,
                'name': 'actions'
            },
            # Operation summary for results
            {
                'type': 'summary',
                'component': child_components['operation_summary'],
                'order': 5,
                'name': 'operation_summary'
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
