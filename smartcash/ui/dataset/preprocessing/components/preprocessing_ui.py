# -*- coding: utf-8 -*-
"""
File: smartcash/ui/dataset/preprocessing/components/preprocessing_ui.py
Description: Main UI components for the preprocessing module using standard container components.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.core.decorators import handle_ui_errors
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import ActionContainer, create_action_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.info_accordion import create_info_accordion
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
from smartcash.ui.dataset.preprocessing.constants import UI_CONFIG, BUTTON_CONFIG, PREPROCESSING_TIPS


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets for the preprocessing module."""
    input_options = create_preprocessing_input_options(config)
    return {
        'components': {
            'input_options': input_options
        },
        'input_options': input_options
    }

def _create_module_summary_content(components: Dict[str, Any]) -> str:
    """Create summary content for the module."""
    return "<p>Pengaturan pra-pemrosesan saat ini dan konfigurasi preset YOLO akan ditampilkan di sini.</p>"

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
def _create_action_buttons() -> Dict[str, Any]:
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
        'action_container': action_container,
        'buttons': buttons
    }

def create_preprocessing_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create preprocessing UI using standard container components."""
    config = config or {}
    ui_components = {}

    # 1. Header
    header_obj = create_header_container(
        title=UI_CONFIG['title'],
        logo_path=UI_CONFIG['logo_path'],
        subtitle=UI_CONFIG['subtitle']
    )
    header_widget = header_obj.container

    # 2. Form
    form_widget = create_preprocessing_input_options()

    # 3. Action Buttons
    action_components = _create_action_buttons()
    action_container = action_components['action_container']
    action_widget = action_container.container
    
    # Get the action container and buttons
    button_widgets = action_components.get('buttons', {})
    
    # Log the buttons we found
    button_ids = list(button_widgets.keys())
    
    # Define button configurations
    button_configs = [
        {'id': 'preprocess', 'text': '🚀 Mulai Preprocessing', 'style': 'primary'},
        {'id': 'check', 'text': '🔍 Check Dataset', 'style': 'info'},
        {'id': 'cleanup', 'text': '🗑️ Cleanup', 'style': 'warning'},
        {'id': 'save', 'text': '💾 Simpan', 'style': 'success'},
        {'id': 'reset', 'text': '🔄 Reset', 'style': 'danger'}
    ]
    
    # Register all buttons
    for btn_cfg in button_configs:
        button_id = btn_cfg['id']
        widget = button_widgets.get(button_id)
        
        if widget is not None:
            # Update button properties
            if hasattr(widget, 'description'):
                widget.description = btn_cfg['text']
            
            # Set button style if available
            if hasattr(widget, 'button_style') and hasattr(widget, 'style'):
                widget.button_style = btn_cfg['style']
            
            # Register the button with its base ID
            ui_components[button_id] = widget
            
            # Set a custom attribute to store the button ID
            setattr(widget, '_button_id', button_id)
    
    # Add the action container to UI components
    ui_components['action_container'] = action_container
    
    # Explicitly expose all buttons in the action container
    if hasattr(action_container, 'buttons') and action_container.buttons:
        for btn_id, btn in action_container.buttons.items():
            if btn_id not in ui_components:
                ui_components[btn_id] = btn
    
    # Log all registered buttons for debugging
    registered_buttons = []
    button_details = {}
    
    for k, v in ui_components.items():
        is_button = (hasattr(v, 'on_click') or 
                    isinstance(v, widgets.Button) or 
                    k in button_widgets or
                    (hasattr(v, 'button_style') and hasattr(v, 'description')))
        
        if is_button:
            registered_buttons.append(k)
            button_details[k] = {
                'description': getattr(v, 'description', 'N/A'),
                'type': type(v).__name__,
                'has_on_click': hasattr(v, 'on_click'),
                'button_id': getattr(v, '_button_id', 'N/A')
            }
    
    # 4. Operation Container
    operation_dict = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name=UI_CONFIG['module_name']
    )
    operation_widget = operation_dict.pop('container', None)
    ui_components.update(operation_dict)

    # 5. Operation Summary (initially hidden, to be placed inside operation_widget)
    operation_summary_obj = create_summary_container(title="Ringkasan Operasi", icon="📊")
    operation_summary_widget = operation_summary_obj.container
    operation_summary_widget.layout.display = 'none'
    ui_components['operation_summary_container'] = operation_summary_widget
    ui_components['operation_summary_updater'] = operation_summary_obj.set_content
    
    # Add summary to the operation container's children
    if operation_widget and isinstance(operation_widget, widgets.Box):
        operation_widget.children = (*operation_widget.children, operation_summary_widget)

    # 6. Footer
    info_box = _create_module_info_box()
    footer_obj = create_footer_container(info_box=info_box)
    footer_widget = footer_obj.container

    # 7. Assemble Main Container using the flexible component list
    component_list = [
        {'name': 'header', 'component': header_widget, 'order': 0},
        {'name': 'form', 'component': form_widget, 'order': 1},
        {'name': 'action', 'component': action_widget, 'order': 2},
        {'name': 'operation', 'component': operation_widget, 'order': 3},
        {'name': 'footer', 'component': footer_widget, 'order': 4},
    ]
    
    main_container_obj = create_main_container(components=component_list)
    ui_components['main_container'] = main_container_obj.container
    ui_components['ui'] = main_container_obj.container  # Alias for display

    return ui_components
