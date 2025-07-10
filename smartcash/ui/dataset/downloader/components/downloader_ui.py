"""
Dataset Downloader UI Module.

This module implements the UI components for the dataset downloader interface,
following the SmartCash UI standardization guidelines.

Features:
- Standardized container structure
- Consistent error handling
- Progress tracking and logging
- Configurable action buttons
- Responsive layout
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable

# Standard container imports
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Import downloader specific components
from .input_options import create_downloader_input_options

# Module configuration constants
UI_CONFIG = {
    'title': "📥 Dataset Downloader",
    'subtitle': "Download dataset Roboflow untuk SmartCash dengan UUID renaming dan validasi otomatis",
    'icon': "📥",
    'module_name': "downloader",
    'parent_module': "dataset",
    'version': "1.0.0"
}

# Button configuration
BUTTON_CONFIG = {
    'download': {
        'text': '📥 Download',
        'style': 'primary',
        'tooltip': 'Download dataset from Roboflow',
        'order': 1
    },
    'check': {
        'text': '🔍 Check',
        'style': 'info',
        'tooltip': 'Check dataset status and integrity',
        'order': 2
    },
    'cleanup': {
        'text': '🗑️ Cleanup',
        'style': 'danger',
        'tooltip': 'Remove dataset files from local storage',
        'order': 3
    }
}

# Validation rules for form fields
VALIDATION_RULES = {
    'workspace': {'required': True, 'min_length': 1},
    'project': {'required': True, 'min_length': 1},
    'version': {'required': True, 'min_length': 1},
    'api_key': {'required': True, 'min_length': 10}
}


@handle_ui_errors(error_component_title="Dataset Downloader UI Creation Error")
def create_downloader_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create and initialize the Dataset Downloader UI components.
    
    This function creates a standardized UI for the dataset downloader module,
    following the SmartCash UI component architecture.
    
    Args:
        config: Optional configuration dictionary to initialize the UI
        **kwargs: Additional keyword arguments passed to component creators
        
    Returns:
        Dict containing all UI components and their references
    """
    if config is None:
        config = {}
    
    # Initialize components dictionary
    components = {
        'ui_initialized': False,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        'config': config.copy()
    }
    
    # 1. Create Header Container
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        status_text="Ready",
        icon=UI_CONFIG['icon']
    )
    
    # 2. Create Form Container
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding="0",
        gap="12px"
    )
    
    # Add input options to form
    input_options = create_downloader_input_options(config)
    form_container['add_item'](input_options, "input_options")
    
    # 3. Create Action Container
    action_container = create_action_container(
        title="Download Actions",
        buttons=[BUTTON_CONFIG['check'], BUTTON_CONFIG['cleanup']],
        primary_button=BUTTON_CONFIG['download'],
        show_save_reset=True,
        alignment="left"
    )
    
    # 4. Create Operation Container
    operation_container = create_operation_container(
        show_progress=True,
        show_logs=True,
        log_module_name=UI_CONFIG['title'],
        log_height="200px"
    )
    
    # 5. Create Footer Container
    from smartcash.ui.info_boxes.download_info import get_download_info
    footer_container = create_footer_container(
        info_items=[get_download_info(open_by_default=False)],
        tips=[
            "💡 Pastikan koneksi internet stabil saat mendownload dataset",
            "🔍 Selalu periksa status dataset sebelum mendownload"
        ]
    )
    
    # 6. Create Main Container
    main_container = create_main_container(
        components=[
            {'component': header_container.container, 'type': 'header'},
            {'component': form_container['container'], 'type': 'form'},
            {'component': action_container['container'], 'type': 'action'},
            {'component': operation_container['container'], 'type': 'operation'},
            {'component': footer_container.container, 'type': 'footer'}
        ],
        container_style={
            'width': '100%',
            'max_width': '1200px',
            'margin': '0 auto',
            'padding': '20px',
            'border': '1px solid #e0e0e0',
            'border_radius': '8px',
            'box_shadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    )
    
    # Store references to all components
    components.update({
        # Main UI components
        'ui': main_container,
        'container': main_container,
        'header_container': header_container.container,
        'form_container': form_container['container'],
        'action_container': action_container['container'],
        'operation_container': operation_container['container'],
        'footer_container': footer_container.container,
        
        # Input components
        'workspace_input': getattr(input_options, 'workspace_input', None),
        'project_input': getattr(input_options, 'project_input', None),
        'version_input': getattr(input_options, 'version_input', None),
        'api_key_input': getattr(input_options, 'api_key_input', None),
        'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        
        # Button references
        'download_button': action_container['primary_button'],
        'check_button': action_container['buttons'].get('check'),
        'cleanup_button': action_container['buttons'].get('cleanup'),
        'save_button': action_container.get('save_button'),
        'reset_button': action_container.get('reset_button'),
        
        # Operation functions
        'log_message': operation_container['log_message'],
        'update_progress': operation_container['update_progress'],
        'show_dialog': operation_container['show_dialog'],
        
        # Mark as initialized
        'ui_initialized': True
    })
    
    return components

# Maintain backward compatibility
create_downloader_ui_components = create_downloader_ui