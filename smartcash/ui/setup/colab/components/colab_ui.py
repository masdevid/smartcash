"""
File: smartcash/ui/setup/colab/components/colab_ui.py
Description: Main UI component for the colab module
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

# Third-party imports
import ipywidgets as widgets

# SmartCash UI components
from smartcash.ui.components import (
    create_action_container,
    create_footer_container,
    create_header_container,
    create_main_container,
    create_operation_container,
)
from smartcash.ui.components.footer_container import PanelConfig, PanelType
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.core.decorators import handle_ui_errors

# Local module imports
from ..constants import UI_CONFIG, BUTTON_CONFIG, COLAB_PHASES

# Local component imports
from .colab_ui_environment import create_environment_container as _create_environment_container
from .colab_ui_form import create_module_form_widgets as _create_module_form_widgets
from .colab_ui_summary import create_module_summary_content as _create_module_summary_content
from .colab_ui_footer import (
    create_module_info_box as _create_module_info_box,
    create_module_tips_box as _create_module_tips_box
)

# Module metadata
MODULE_METADATA = {
    'module_name': 'colab',
    'parent_module': 'setup',
    'ui_initialized': True,
    'config': {}
}

# Re-export constants for backward compatibility
UI_CONFIG = UI_CONFIG  # type: ignore
BUTTON_CONFIG = BUTTON_CONFIG  # type: ignore


def _create_colab_ui_components(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create and configure all UI components for the colab module.
    
    The components are created in the following order:
    1. Header section with title and subtitle
    2. Form widgets (colab-specific configuration)
    3. Action buttons (setup, save, reset)
    4. Operation container for progress and logs
    5. Environment container for system information
    6. Footer with info accordion
    7. Main container assembly
    
    Args:
        config: Configuration dictionary for the UI components
        
    Returns:
        Tuple of (ui_components, widgets) dictionaries
    """
    # Initialize component containers
    ui_components = {}
    widgets_dict = {}
    
    # 1. Create header
    header = _create_header()
    ui_components['header_container'] = header
    
    # 2. Create form widgets
    form_widgets = _create_module_form_widgets(config)
    ui_components['form_container'] = {
        'container': form_widgets.get('form_ui') or form_widgets.get('container'),
        'widgets': form_widgets
    }
    widgets_dict.update(form_widgets)
    
    # 3. Create action buttons
    action_container = _create_action_buttons()
    ui_components['action_container'] = action_container
    
    # Extract button references for proper registration
    # Only setup_button (primary one-click operation), save_button, and reset_button
    buttons_dict = action_container.get('buttons', {})
    colab_setup_button = buttons_dict.get('colab_setup')
    ui_components['setup_button'] = colab_setup_button
    ui_components['primary_button'] = colab_setup_button  # Same button, multiple references
    ui_components['colab_setup'] = colab_setup_button     # Original ID reference
    
    # Extract save/reset buttons from action container instance
    action_container_instance = action_container.get('action_container')
    if action_container_instance:
        ui_components['save_button'] = getattr(action_container_instance, 'save_button', None)
        ui_components['reset_button'] = getattr(action_container_instance, 'reset_button', None)
    
    # Extract phase management methods for setup button phases
    ui_components['set_phase'] = action_container.get('set_phase')
    ui_components['set_phases'] = action_container.get('set_phases')
    ui_components['enable_all'] = action_container.get('enable_all')
    ui_components['disable_all'] = action_container.get('disable_all')
    
    # 4. Create summary container
    summary_content = "<div style='padding: 10px; width: 100%;'>Colab setup summary will appear here...</div>"
    summary_container = create_summary_container(
        theme='default',
        title='Colab Setup Summary',
        icon='📊'
    )
    summary_container.set_content(summary_content)
    ui_components['summary_container'] = {
        'container': summary_container,
        'content': summary_content,
        'widgets': {}
    }
    
    # 5. Create operation container
    operation_container = _create_operation_container()
    ui_components['operation_container'] = operation_container
    
    # 6. Create environment container
    environment_container = _create_environment_container(config)
    ui_components['environment_container'] = environment_container
    
    # 7. Create footer with info box
    footer_container = _create_footer()
    ui_components['footer_container'] = footer_container
    
    # 8. Assemble main container
    main_container = _assemble_main_container(
        header_container=header,
        form_container=form_widgets.get('form_ui') or form_widgets.get('container'),
        action_container=action_container,
        summary_container=summary_container.container,
        operation_container=operation_container,
        environment_container=environment_container,
        footer_container=footer_container
    )
    
    # Update component references
    ui_components.update({
        'main_container': main_container,
        'main_layout': main_container.container,
        'summary_container': summary_container  # Keep reference to summary container
    })
    
    # Add summary container to widgets dictionary
    widgets_dict['summary_container'] = summary_container
    widgets_dict['form_container'] = form_widgets.get('form_ui') or form_widgets.get('container')
    
    return ui_components, widgets_dict


def _create_header() -> Any:
    """Create the header section with title and subtitle."""
    return create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon=UI_CONFIG['icon'],
        status_message="Siap untuk pengaturan lingkungan",
        status_type="info",
        show_status_panel=True
    )


def _create_action_buttons() -> Dict[str, Any]:
    """Create action buttons container with primary setup button and phases."""
    # Single primary button for all colab setup phases
    action_buttons = [
        {
            'id': 'colab_setup',
            'text': '🚀 Mulai Setup',
            'style': 'primary',
            'tooltip': 'One-click setup untuk semua fase lingkungan Colab',
            'order': 1
        }
    ]
    
    return create_action_container(
        buttons=action_buttons,
        title="🚀 Colab Environment Setup",
        container_margin="15px 0 5px 0",
        show_save_reset=True,  # Include save/reset buttons
        phases=COLAB_PHASES  # Pass phases for sequential operations
    )


def _create_operation_container() -> Dict[str, Any]:
    """Create operation container for logs and progress."""
    return create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="Colab",
        # Remove restrictive namespace filter to allow all logs through
        # log_namespace_filter='colab',  # This was blocking logs
        progress_levels='dual'
    )


def _create_footer() -> Any:
    """Create footer with info accordion."""
    return create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="📚 Colab Setup Guide",
                content=_create_module_info_box().value,
                style="info",
                flex="1",
                min_width="50%",
                open_by_default=False
            ),
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="💡 Setup Tips",
                content=_create_module_tips_box().value,
                style="info",
                flex="1",
                min_width="50%",
                open_by_default=False
            )
        ],
        style={
            'border_top': '1px solid #e0e0e0',
            'background': '#f9f9f9',
            'margin_top': '15px',
            'padding': '10px'
        }
    )


def _assemble_main_container(
    header_container: Any,
    form_container: Any,
    action_container: Dict[str, Any],
    summary_container: Any,
    operation_container: Dict[str, Any],
    environment_container: Any,
    footer_container: Any
) -> Any:
    """Assemble all components into the main container."""
    components = [
        {'type': 'header', 'component': header_container.container, 'order': 0, 'name': 'header'},
        {'type': 'form', 'component': form_container, 'order': 1, 'name': 'form'},
        {'type': 'action', 'component': action_container['container'], 'order': 2, 'name': 'actions'},
        {'type': 'summary', 'component': summary_container, 'order': 3, 'name': 'summary', 'visible': False},
        {'type': 'operation', 'component': operation_container['container'], 'order': 4, 'name': 'operations'},
        {'type': 'environment', 'component': environment_container, 'order': 5, 'name': 'environment'},
        {'type': 'footer', 'component': footer_container.container, 'order': 6, 'name': 'footer'}
    ]
    
    return create_main_container(
        components=components,
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='0 10px 20px 10px',
        align_items='stretch'
    )


@handle_ui_errors(error_component_title="Colab UI Creation Error")
def create_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create main colab UI using container-based architecture.
    
    Features:
    - 🏗️ Container-based architecture with core inheritance
    - 🎨 Preserved original forms and unique styling
    - 📊 Added summary_container component
    - 🔄 Standardized button and action handling
    - 📝 Comprehensive logging and progress tracking
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components and metadata
    """
    config = config or {}
    
    # Create UI components
    ui_components, widgets_dict = _create_colab_ui_components(config)
    
    # Create container variables in function scope (required by validator)
    header_container = ui_components['header_container']
    form_container = ui_components['form_container']
    action_container = ui_components['action_container']
    operation_container = ui_components['operation_container']
    footer_container = ui_components['footer_container']
    
    # Preserve the button references from the original ui_components
    # Include all button references for proper handler registration
    button_refs = {
        'setup_button': ui_components.get('setup_button'),
        'primary_button': ui_components.get('primary_button'),
        'colab_setup': ui_components.get('colab_setup'),
        'save_button': ui_components.get('save_button'),
        'reset_button': ui_components.get('reset_button'),
        'set_phase': ui_components.get('set_phase'),
        'set_phases': ui_components.get('set_phases'),
        'enable_all': ui_components.get('enable_all'),
        'disable_all': ui_components.get('disable_all')
    }
    
    # Create UI components dictionary
    ui_components_final = {
        'module_name': 'colab',
        'parent_module': 'setup',
        'ui_initialized': True,
        'config': config or {},
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        'main_container': ui_components.get('main_layout'),  # Use the actual widget, not the MainContainer object
        'environment_container': ui_components.get('environment_container'),
        **widgets_dict,
        **button_refs,  # Include all button references and phase methods
        # Add ui_components as a flat dictionary with all components
        'ui_components': {
            'header': header_container,
            'form': form_container,
            'actions': action_container,
            'operation': operation_container,
            'footer': footer_container,
            'environment': ui_components.get('environment_container'),
            'main': ui_components.get('main_layout'),  # Use the actual widget, not the MainContainer object
            'widgets': widgets_dict,
            'metadata': MODULE_METADATA,
            'config': config
        }
    }
    
    return ui_components_final
    
