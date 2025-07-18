"""
File: smartcash/ui/dataset/augmentation/components/augmentation_ui.py
Description: Main UI component for the augmentation module
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
from ..constants import BUTTON_CONFIG, UI_CONFIG

# Local component imports
from .form_widgets import create_module_form_widgets
from .info_accordion import create_info_accordion

# Module metadata
MODULE_METADATA = {
    'module_name': 'augment',
    'parent_module': 'dataset',
    'ui_initialized': True,
    'config': {}
}

# Re-export constants for backward compatibility
UI_CONFIG = UI_CONFIG  # type: ignore
BUTTON_CONFIG = BUTTON_CONFIG  # type: ignore


def _create_augment_ui_components(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create and configure all UI components for the augment module.
    
    The components are created in the following order:
    1. Header section with title and subtitle
    2. Form widgets (basic options, advanced options, augmentation types)
    3. Action buttons (preview, augment, check, cleanup)
    4. Operation container for progress and logs
    5. Footer with info accordion
    6. Main container assembly
    
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
    form_widgets = create_module_form_widgets(config)
    ui_components['form_container'] = {
        'container': form_widgets['container'],
        'widgets': form_widgets['widgets']
    }
    widgets_dict.update(form_widgets['widgets'])
    
    # Extract generate button from preview widget for proper registration
    preview_widget = form_widgets['widgets'].get('preview_widget')
    if preview_widget and isinstance(preview_widget, dict):
        preview_widgets = preview_widget.get('widgets', {})
        generate_button = preview_widgets.get('generate')
        if generate_button:
            ui_components['generate'] = generate_button
            widgets_dict['generate'] = generate_button
    
    # 3. Create action buttons
    action_container = _create_action_buttons()
    ui_components['action_container'] = action_container
    
    # Extract individual operation buttons from the buttons dictionary
    buttons_dict = action_container.get('buttons', {})
    ui_components['augment'] = buttons_dict.get('augment')
    ui_components['status'] = buttons_dict.get('status')
    ui_components['cleanup'] = buttons_dict.get('cleanup')
    
    # Extract save/reset buttons from action container instance
    action_container_instance = action_container.get('action_container')
    if action_container_instance:
        save_btn = getattr(action_container_instance, 'save_button', None)
        reset_btn = getattr(action_container_instance, 'reset_button', None)
        ui_components['save'] = save_btn
        ui_components['reset'] = reset_btn
    
    
    # 4. Create summary container
    summary_content = "<div style='padding: 10px; width: 100%;'>Augmentation summary will appear here...</div>"
    summary_container = create_summary_container(
        theme='default',
        title='Augmentation Summary',
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
    
    # 5. Create footer with info box
    footer_container = _create_footer()
    ui_components['footer_container'] = footer_container
    
    # 6. Assemble main container
    main_container = _assemble_main_container(
        header_container=header,
        form_container=form_widgets['container'],
        action_container=action_container,
        summary_container=summary_container.container,
        operation_container=operation_container,
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
    widgets_dict['form_container'] = form_widgets['container']
    
    return ui_components, widgets_dict


def _create_header() -> Any:
    """Create the header section with title and subtitle."""
    return create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='🔄'  # Refresh emoji for augmentation
    )


def _create_action_buttons() -> Dict[str, Any]:
    """Create action buttons container with all operation buttons."""
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
        title="🚀 Augmentation Operations",
        container_margin="15px 0 5px 0",
        show_save_reset=True  # Include save/reset buttons
    )


def _create_operation_container() -> Dict[str, Any]:
    """Create operation container for logs and progress with dual progress tracking."""
    return create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name="Augmentation",
        # log_namespace_filter='augmentation',  # Temporarily disabled
        log_height="150px",
        log_entry_style='compact',
        collapsible=True,
        collapsed=False
    )


def _create_footer() -> Any:
    """Create footer with info accordion."""
    return create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="📚 Augmentation Guide",
                content=create_info_accordion(),
                style="info",
                flex="1",
                min_width="100%",
                open_by_default=True
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
    footer_container: Any
) -> Any:
    """Assemble all components into the main container."""
    components = [
        {'type': 'header', 'component': header_container.container, 'order': 0, 'name': 'header'},
        {'type': 'form', 'component': form_container, 'order': 1, 'name': 'form'},
        {'type': 'action', 'component': action_container['container'], 'order': 2, 'name': 'actions'},
        {'type': 'summary', 'component': summary_container, 'order': 3, 'name': 'summary', 'visible': False},
        {'type': 'operation', 'component': operation_container['container'], 'order': 4, 'name': 'operations'},
        {'type': 'footer', 'component': footer_container.container, 'order': 5, 'name': 'footer'}
    ]
    
    return create_main_container(
        components=components,
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='0 10px 20px 10px',
        align_items='stretch'
    )


@handle_ui_errors(error_component_title="Augment UI Creation Error")
def create_augment_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create main augment UI using container-based architecture.
    
    Features:
    - 🏗️ Container-based architecture with core inheritance
    - 🎨 Preserved original forms and unique styling
    # - 📊 Added summary_container component
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
    ui_components, widgets_dict = _create_augment_ui_components(config)
    
    # Create container variables in function scope (required by validator)
    header_container = ui_components['header_container']
    form_container = ui_components['form_container']
    action_container = ui_components['action_container']
    operation_container = ui_components['operation_container']
    footer_container = ui_components['footer_container']
    
    # Preserve the button references from the original ui_components
    button_refs = {
        'augment': ui_components.get('augment'),
        'status': ui_components.get('status'),
        'cleanup': ui_components.get('cleanup'),
        'save': ui_components.get('save'),
        'reset': ui_components.get('reset'),
        'generate': ui_components.get('generate')
    }
    
    # Create UI components dictionary
    ui_components_final = {
        'module_name': 'augment',
        'parent_module': 'dataset',
        'ui_initialized': True,
        'config': config or {},
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        'main_container': ui_components.get('main_layout'),  # Use the actual widget, not the MainContainer object
        **widgets_dict,
        **button_refs,  # Include all button references
        # Add ui_components as a flat dictionary with all components
        'ui_components': {
            'header': header_container,
            'form': form_container,
            'actions': action_container,
            'operation': operation_container,
            'footer': footer_container,
            'main': ui_components.get('main_layout'),  # Use the actual widget, not the MainContainer object
            'widgets': widgets_dict,
            'metadata': MODULE_METADATA,
            'config': config
        }
    }
    
    return ui_components_final