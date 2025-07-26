"""
File: smartcash/ui/dataset/augmentation/components/augmentation_ui.py
Description: Optimized main UI component for the augmentation module
"""

from typing import Any, Dict, Optional
import ipywidgets as widgets

# SmartCash UI components
from smartcash.ui.components import (
    create_action_container,
    create_header_container,
    create_main_container,
    create_operation_container,
)
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.core.decorators import handle_ui_errors

# Local module imports
from ..constants import BUTTON_CONFIG, UI_CONFIG
from .form_widgets import create_module_form_widgets


def _create_augment_ui_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create and configure all UI components for the augment module.
    
    Args:
        config: Configuration dictionary for the UI components
        
    Returns:
        Dictionary containing all UI components
    """
    # 1. Create header
    header = _create_header()
    
    # 2. Create form widgets
    form_widgets = create_module_form_widgets(config)
    
    # 3. Create action buttons
    action_container = _create_action_buttons()
    
    # 4. Create summary container
    summary_container = create_summary_container(
        theme='default',
        title='Ringkasan Augmentasi',
        icon='📊'
    )
    summary_content = "<div style='padding: 10px; width: 100%;'>Ringkasan augmentasi akan muncul di sini...</div>"
    summary_container.set_content(summary_content)
    
    # 5. Create operation container
    operation_container = _create_operation_container()
    
    # 6. Assemble main container
    main_container = _assemble_main_container(
        header_container=header,
        form_container=form_widgets['container'],
        action_container=action_container,
        summary_container=summary_container.container,
        operation_container=operation_container,
    )
    
    # Extract generate button from preview widget
    generate_button = None
    preview_widget = form_widgets['widgets'].get('preview_widget')
    if preview_widget and isinstance(preview_widget, dict):
        preview_widgets = preview_widget.get('widgets', {})
        generate_button = preview_widgets.get('generate')
    
    # Return organized components
    return {
        'main_container': main_container.container,
        'header_container': header,
        'form_container': form_widgets,
        'action_container': action_container,
        'operation_container': operation_container,
        'summary_container': summary_container,
        'generate': generate_button,
        **form_widgets['widgets'],
        **action_container.get('buttons', {})
    }


def _create_header() -> Any:
    """Create the header section with title and subtitle."""
    return create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='🔄',  # Refresh emoji for augmentation
        show_environment=True,
        environment='local',
        config_path='augmentation_config.yaml'
    )


def _create_action_buttons() -> Dict[str, Any]:
    """Create container with augment/status/cleanup action buttons."""
    action_buttons = [
        {
            'id': btn_id,
            'text': BUTTON_CONFIG[btn_id]['text'],
            'style': BUTTON_CONFIG[btn_id]['style'],
            'tooltip': BUTTON_CONFIG[btn_id]['tooltip'],
            'order': BUTTON_CONFIG[btn_id]['order']
        }
        for btn_id in ['augment', 'status', 'cleanup']
    ]
    
    return create_action_container(
        buttons=action_buttons,
        title="🚀 Operasi Augmentasi",
        container_margin="15px 0 5px 0",
        show_save_reset=True  # Include save/reset buttons
    )


def _create_operation_container() -> Dict[str, Any]:
    """Create container for logs and dual progress tracking."""
    return create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name="Augmentation",
        log_height="150px",
        collapsible=True,
        collapsed=False
    )



def _assemble_main_container(
    header_container: Any,
    form_container: Any,
    action_container: Dict[str, Any],
    summary_container: Any,
    operation_container: Dict[str, Any],
) -> Any:
    """Assemble all components into the main container."""
    components = [
        {'type': 'header', 'component': header_container.container, 'order': 0, 'name': 'header'},
        {'type': 'form', 'component': form_container, 'order': 1, 'name': 'form'},
        {'type': 'action', 'component': action_container['container'], 'order': 2, 'name': 'actions'},
        {'type': 'operation', 'component': operation_container['container'], 'order': 3, 'name': 'operations'},
        {'type': 'summary', 'component': summary_container, 'order': 4, 'name': 'summary', 'visible': False},
    ]
    
    return create_main_container(
        components=components,
    )


@handle_ui_errors(error_component_title="Augment UI Creation Error")
def create_augment_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create and configure the augmentation UI with all components.
    
    Returns:
        Dict containing all UI components
    """
    config = config or {}
    
    # Create all UI components
    ui_components = _create_augment_ui_components(config)
    
    # Add nested UI components structure for compatibility
    ui_components['ui_components'] = {
        'header': ui_components['header_container'],
        'form': ui_components['form_container'],
        'actions': ui_components['action_container'],
        'operation': ui_components['operation_container'],
        'main': ui_components['main_container'],
        'widgets': {k: v for k, v in ui_components.items() if k.endswith('_widget') or k == 'generate'},
        'config': config
    }
    
    # Add module metadata
    ui_components.update({
        'module_name': 'augment',
        'parent_module': 'dataset',
        'ui_initialized': True,
        'config': config
    })
    
    return ui_components