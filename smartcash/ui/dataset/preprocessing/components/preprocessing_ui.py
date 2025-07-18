# -*- coding: utf-8 -*-
"""
File: smartcash/ui/dataset/preprocessing/components/preprocessing_ui.py
Description: Main UI components for the preprocessing module using standard container components.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import ipywidgets as widgets

# Core UI components
from smartcash.ui.components import (
    create_main_container,
    create_header_container,
    ActionContainer,
    create_footer_container,
    create_operation_container,
    create_summary_container
)

# Core UI utilities
from smartcash.ui.core.decorators import handle_ui_errors
from smartcash.ui.core.errors.enums import ErrorLevel

# Module-specific imports
from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
from smartcash.ui.dataset.preprocessing.constants import UI_CONFIG, PREPROCESSING_TIPS


def _create_module_info_box() -> widgets.Widget:
    """Create an info box with module documentation."""
    tips_html = ''.join([f"<li>{tip}</li>" for tip in PREPROCESSING_TIPS])
    return widgets.HTML(
        value=f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8; margin: 10px 0;">
            <h5>ℹ️ Tips Pra-pemrosesan</h5>
            <ul>{tips_html}</ul>
            <div style="margin-top: 10px; padding: 8px; background-color: #e7f3ff; border-radius: 4px;">
                <strong>Preset YOLO:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li><strong>yolov5s/m:</strong> 640x640 - Standar untuk training cepat</li>
                    <li><strong>yolov5l:</strong> 832x832 - Akurasi lebih tinggi, lebih lambat</li>
                    <li><strong>yolov5x:</strong> 1024x1024 - Akurasi maksimum</li>
                </ul>
            </div>
        </div>
        """
    )

@handle_ui_errors(
    error_component_title=f"{UI_CONFIG['module_name']} Error",
    level=ErrorLevel.ERROR,
    show_dialog=True
)
def create_action_buttons() -> Dict[str, Any]:
    """Create action buttons for preprocessing operations using ActionContainer.
    
    Returns:
        Dictionary containing the action container and button widgets
    """
    # Create action container with save/reset buttons enabled
    action_container = ActionContainer(
        show_save_reset=True,
        container_margin='12px 0 0 0'
    )
    
    # Add action buttons with proper ordering and styling
    action_container.add_button(
        button_id='preprocess',
        text='🚀 Mulai Preprocessing',
        style='success',
        tooltip='Mulai proses preprocessing data',
        order=1
    )
    
    action_container.add_button(
        button_id='check',
        text='🔍 Check Dataset',
        style='info',
        tooltip='Periksa dataset sebelum preprocessing',
        order=2
    )
    
    action_container.add_button(
        button_id='cleanup',
        text='🗑️ Cleanup',
        style='danger',
        tooltip='Bersihkan hasil preprocessing',
        order=3
    )
    
    # Get references to all buttons for easy access
    buttons = {
        'preprocess': action_container.get_button('preprocess'),
        'check': action_container.get_button('check'),
        'cleanup': action_container.get_button('cleanup'),
        'save': action_container.save_button,
        'reset': action_container.reset_button
    }
    
    return {
        'container': action_container.container,
        'buttons': buttons
    }


def create_preprocessing_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create preprocessing UI using standard container components.
    
    Returns:
        Dictionary containing all UI components and their references
    """
    config = config or {}
    ui_components = {}

    # 1. Header
    header = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='🧹'  # Broom emoji for preprocessing
    )
    header_container = header.container

    # 2. Form
    form_container = create_preprocessing_input_options()

    # Create action buttons
    action_components = create_action_buttons()
    action_container = action_components['container']
    action_buttons = action_components['buttons']
    ui_components['check_button'] = action_buttons['check']
    ui_components['preprocess_button'] = action_buttons['preprocess']
    ui_components['cleanup_button'] = action_buttons['cleanup']
    ui_components['save_button'] = action_buttons['save']
    ui_components['reset_button'] = action_buttons['reset']
    
    # 4. Operation Container with dual progress for preprocessing operations
    operation_dict = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name=UI_CONFIG['module_name'],
        # log_namespace_filter='preprocessing',  # Temporarily disabled
        log_height="150px",
        log_entry_style='compact',
        collapsible=True,
        collapsed=False
    )
    
    # Get the operation container and update UI components
    operation_container = operation_dict.pop('container', None)
    
    # Add operation container and its components to ui_components
    if operation_container is not None:
        ui_components['operation_container'] = operation_container
    
    # Update with operation dictionary components
    if operation_dict:
        for key, value in operation_dict.items():
            if value is not None and key not in ui_components:
                ui_components[key] = value

    # 5. Operation Summary (initially hidden, to be placed inside operation_container)
    operation_summary_obj = create_summary_container(title="Ringkasan Operasi", icon="📊")
    operation_summary_widget = operation_summary_obj.container
    operation_summary_widget.layout.display = 'none'
    ui_components['operation_summary_container'] = operation_summary_widget
    ui_components['operation_summary_updater'] = operation_summary_obj.set_content
    
    # Add summary to the operation container's children
    if operation_container and isinstance(operation_container, widgets.Box):
        operation_container.children = (*operation_container.children, operation_summary_widget)

    # 6. Footer
    info_box = _create_module_info_box()
    footer = create_footer_container(info_box=info_box)
    footer_container = footer.container

    # 7. Assemble Main Container using the flexible component list
    component_list = [
        {'name': 'header', 'component': header_container, 'order': 0},
        {'name': 'form', 'component': form_container, 'order': 1},
        {'name': 'action', 'component': action_container, 'order': 2},
        {'name': 'operation', 'component': operation_container, 'order': 3},
        {'name': 'footer', 'component': footer_container, 'order': 4},
    ]
    
    main_container = create_main_container(components=component_list)
    
    # Ensure all main container components are properly exposed
    ui_components['main_container'] = main_container.container
    ui_components['main_layout'] = main_container.container
    ui_components['ui'] = main_container.container  # Alias for display
    
    # Make sure all components are properly exposed
    if hasattr(main_container, 'components'):
        for comp in main_container.components:
            if 'name' in comp and 'component' in comp:
                ui_components[comp['name']] = comp['component']

    return ui_components
