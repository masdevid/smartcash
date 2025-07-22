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
    
    Note: This function is optimized to only create components when needed
    and caches them for better performance.
    
    Args:
        config: Configuration dictionary for the UI components
        
    Returns:
        Tuple of (ui_components, widgets) dictionaries
    """
    # Initialize component containers
    ui_components = {}
    widgets_dict = {}
    
    # 1. Create header - only create if not already in cache
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
    
    # Extract button references with consistent access pattern
    buttons_dict = action_container.get('buttons', {})
    
    # Store button references with consistent naming
    ui_components['colab_setup'] = buttons_dict.get('colab_setup')
    ui_components['save'] = buttons_dict.get('save')
    ui_components['reset'] = buttons_dict.get('reset')
    
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
        'main_container': main_container.container,  # Use the actual widget, not the MainContainer object
        'summary_container': summary_container  # Keep reference to summary container
    })
    
    # Add summary container to widgets dictionary
    widgets_dict['summary_container'] = summary_container
    widgets_dict['form_container'] = form_widgets.get('form_ui') or form_widgets.get('container')
    
    return ui_components, widgets_dict


def _create_header(): return create_header_container(title=UI_CONFIG['title'], subtitle=UI_CONFIG['subtitle'], icon=UI_CONFIG['icon'], show_environment=True, environment='colab', config_path='colab_config.yaml')

def _create_action_buttons():
    return create_action_container(
        buttons=[{'id': 'colab_setup', 'text': '🚀 Mulai Setup', 'style': 'primary', 'tooltip': 'One-click setup untuk semua fase lingkungan Colab', 'order': 1}],
        title="🚀 Colab Environment Setup",
        container_margin="15px 0 5px 0",
        show_save_reset=True,
        phases=COLAB_PHASES
    )

def _create_operation_container():
    return create_operation_container(show_progress=True, show_dialog=True, show_logs=True, log_module_name="Colab", progress_levels='dual')

def _create_footer():
    return create_footer_container(panels=[
        PanelConfig(panel_type=PanelType.INFO_ACCORDION, title="📚 Colab Setup Guide", content=_create_module_info_box().value, style="info", flex="1", min_width="50%", open_by_default=False),
        PanelConfig(panel_type=PanelType.INFO_ACCORDION, title="💡 Setup Tips", content=_create_module_tips_box().value, style="info", flex="1", min_width="50%", open_by_default=False)
    ])

def _assemble_main_container(header_container, form_container, action_container, summary_container, operation_container, environment_container, footer_container):
    return create_main_container(
        header_container=header_container.container,
        form_container=form_container,
        action_container=action_container['container'],
        operation_container=operation_container['container'],
        footer_container=footer_container.container
    )


_ui_components_cache = None

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
    - ⚡ Optimized with component caching
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components and metadata
    """
    global _ui_components_cache
    
    # Return cached components if available and no config changes
    if _ui_components_cache is not None and not config:
        return _ui_components_cache
        
    config = config or {}
    
    # Create UI components
    ui_components, widgets_dict = _create_colab_ui_components(config)
    
    # Cache the components for future use
    _ui_components_cache = ui_components
    
    # Create container variables in function scope (required by validator)
    header_container = ui_components['header_container']
    form_container = ui_components['form_container']
    action_container = ui_components['action_container']
    operation_container = ui_components['operation_container']
    footer_container = ui_components['footer_container']
    
    # Create UI components dictionary matching BaseUIModule expectations
    ui_components_final = {
        'main_container': ui_components.get('main_container'),  # Use standardized main_container key
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        'environment_container': ui_components.get('environment_container'),
        'colab_setup': ui_components.get('colab_setup'),
        'save': ui_components.get('save'),
        'reset': ui_components.get('reset'),
        'widgets': widgets_dict,
        'metadata': MODULE_METADATA,
        'module_name': 'colab',
        'parent_module': 'setup',
        'ui_initialized': True,
        'config': config or {}
    }
    
    return ui_components_final
    