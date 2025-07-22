# -*- coding: utf-8 -*-
"""
File: smartcash/ui/dataset/preprocessing/components/preprocessing_ui.py
Description: Main UI components for the preprocessing module using standard container components.
"""

from typing import Dict, Any, Optional

# Standard container imports (following successful patterns from downloader/evaluation modules)
from smartcash.ui.components import (
    create_header_container, 
    create_form_container, 
    create_action_container, 
    create_operation_container, 
    create_footer_container, 
    create_main_container
)
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.form_container import LayoutType
from smartcash.ui.core.decorators import handle_ui_errors

# Module-specific imports
from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
from smartcash.ui.dataset.preprocessing.constants import UI_CONFIG, PREPROCESSING_TIPS


def _create_module_info_box() -> str:
    """Create info box content for the footer container."""
    tips_html = ''.join([f"<li>{tip}</li>" for tip in PREPROCESSING_TIPS])
    return f"""
    <div style="padding: 10px;">
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


@handle_ui_errors(error_component_title="Preprocessing UI Creation Error")
def create_preprocessing_ui_components(
    config: Optional[Dict[str, Any]] = None, 
    **kwargs
) -> Dict[str, Any]:
    """Create preprocessing UI using standard container components.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments including module_config
        
    Returns:
        Dictionary containing all UI components and their references
    """
    if config is None:
        config = {}
        
    # Merge module_config into config if provided
    if 'module_config' in kwargs:
        config.update(kwargs['module_config'])
    
    # Initialize components dictionary
    components = {
        'ui_initialized': False,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG.get('parent_module', 'dataset'),
        'version': UI_CONFIG.get('version', '1.0'),
        'config': config.copy()
    }
    
    # 1. Create header with title and subtitle
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='🧹',  # Broom emoji for preprocessing
        show_environment=True,
        environment='local',  # Default environment
        config_path='preprocessing_config.yaml'
    )
    components['header_container'] = header_container
    
    # 2. Create Form Container
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding="0",
        gap="12px"
    )
    
    # Add input options to form
    input_options = create_preprocessing_input_options()
    form_container['add_item'](input_options, "input_options")
    
    # 3. Create Action Container (following downloader pattern)
    buttons = [
        {
            'button_id': 'preprocess',
            'text': '🚀 Mulai Preprocessing',
            'style': 'success',
            'tooltip': 'Mulai proses preprocessing data',
            'order': 1
        },
        {
            'button_id': 'check',
            'text': '🔍 Check Dataset',
            'style': 'info',
            'tooltip': 'Periksa dataset sebelum preprocessing',
            'order': 2
        },
        {
            'button_id': 'cleanup',
            'text': '🗑️ Cleanup',
            'style': 'danger',
            'tooltip': 'Bersihkan hasil preprocessing',
            'order': 3
        }
    ]
    
    # Create action container with save/reset buttons as requested
    action_container = create_action_container(
        title="🧹 Preprocessing Operations",
        buttons=buttons,
        show_save_reset=True
    )
    
    # 4. Create Operation Container with dual progress tracker for preprocessing operations
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name=UI_CONFIG['module_name'],
        log_namespace_filter='preprocessing',  # Enable namespace filtering
        log_height="150px",
        log_entry_style='compact',
        collapsible=True,
        collapsed=False
    )
    
    # Get progress tracker and log accordion from operation container
    progress_tracker = operation_container.get('progress_tracker')
    log_accordion = operation_container.get('log_accordion')
    
    # Initialize progress tracker if available
    if progress_tracker and hasattr(progress_tracker, 'initialize'):
        progress_tracker.initialize()
        # Start with the first step if the progress tracker supports it
        if hasattr(progress_tracker, 'start'):
            progress_tracker.start()
    
    # 5. Create Summary Container  
    summary_container = create_summary_container(
        theme='default',
        title='Preprocessing Summary',
        icon='📊'
    )
    summary_content = "<div style='padding: 10px; width: 100%;'>Preprocessing summary will appear here...</div>"
    summary_container.set_content(summary_content)
    
    # 6. Create Footer Container
    footer_container = create_footer_container(
        info_items=[{
            'title': '💡 Tips Preprocessing',
            'content': _create_module_info_box(),
            'style': 'info',
            'open_by_default': False
        }],
        tips=[
            "💡 Pastikan dataset sudah dalam format yang benar sebelum preprocessing",
            "🔍 Selalu periksa hasil preprocessing sebelum melanjutkan ke training"
        ]
    )
    
    # Extract and validate action buttons with consistent naming
    action_buttons = action_container.get('buttons', {})
    
    # Map button IDs to their references using normalized IDs
    button_components = {}
    for button_id in ['preprocess', 'check', 'cleanup', 'save', 'reset']:
        button_ref = action_buttons.get(button_id)
        if button_ref is not None:
            button_components[button_id] = button_ref
    
    # 6. Create components dictionary with all UI elements
    components = {
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,  # Full action container with all methods
        'operation_container': operation_container,  # Full operation container dict for logging
        'footer_container': footer_container,
        'ui_initialized': True,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG.get('parent_module', 'dataset'),
        'version': UI_CONFIG.get('version', '1.0'),
        # Add direct button references
        **button_components
    }
    
    # 7. Create Main Container with all components
    main_container = create_main_container(
        components=[
            {'component': header_container.container, 'type': 'header'},
            {'component': form_container['container'], 'type': 'form'},
            {'component': action_container['container'], 'type': 'action'},
            {'component': summary_container.container, 'type': 'summary'},
            {'component': operation_container['container'], 'type': 'operation'},
            {'component': footer_container.container, 'type': 'footer'}
        ]
    )
    
    # 8. Create the final UI components dictionary with proper null checks
    
    # Safely get operation container components with proper fallbacks
    progress_tracker = (
        operation_container.get('progress_tracker') 
        if operation_container and hasattr(operation_container, 'get') 
        else None
    )
    
    # Get other operation container components
    log_accordion = operation_container.get('log_accordion') if operation_container else None
    update_progress = operation_container.get('update_progress') if operation_container else None
    show_dialog = operation_container.get('show_dialog') if operation_container else None
    
    # Use consistent modern naming (no backward compatibility aliases)
    progress_bar = progress_tracker  # Keep this as it's still used
    # log_output is OBSOLETE - use operation_container['log'] for logging
    
    # Get status text from progress tracker if available
    status_text = None
    if progress_tracker and hasattr(progress_tracker, 'get_status_text'):
        status_text = progress_tracker.get_status_text()
    
    # Create the final components dictionary
    components = {
        'main_container': main_container.container,  # Use the actual widget, not the MainContainer object
        'ui_components': {
            'header': header_container,
            'form': form_container,
            'action': action_container,
            'summary': summary_container,
            'operation': operation_container,
            'footer': footer_container
        },
        'ui_initialized': True,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG.get('parent_module', 'dataset'),
        'version': UI_CONFIG.get('version', '1.0'),
        'config': config,
        'main_container': main_container.container,
        
        # Add container references
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'summary_container': summary_container,
        'operation_container': operation_container,  # Store full container dict for logging
        'footer_container': footer_container,
        
        # Add all button references (using consistent naming)
        'preprocess': action_buttons.get('preprocess'),
        'check': action_buttons.get('check'),
        'cleanup': action_buttons.get('cleanup'),
        'save': action_buttons.get('save'),
        'reset': action_buttons.get('reset'),
        
        # Add form widgets - accessed as attributes of input_options
        'input_options': input_options,
        
        # Add operation widgets with the same instances as above
        'progress_bar': progress_bar,
        'status_text': status_text,
        'log_accordion': log_accordion,  # Use consistent naming
        'update_progress': update_progress,
        'show_dialog': show_dialog
    }
    
    return components