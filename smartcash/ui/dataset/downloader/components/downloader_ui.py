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

from typing import Dict, Any, Optional

# Standard container imports
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.decorators import handle_ui_errors

# Global UI handler removed - functionality moved to UIModule pattern

# Import downloader specific components
from .input_options import create_downloader_input_options
from smartcash.ui.dataset.downloader.constants import (
    UI_CONFIG,
    BUTTON_CONFIG,
    VALIDATION_RULES
)

# Re-export constants to maintain backward compatibility and satisfy validation
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG
VALIDATION_RULES = VALIDATION_RULES


@handle_ui_errors(error_component_title="Dataset Downloader UI Creation Error")
def create_downloader_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        status_message="Ready",
        status_type="info",
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
    # Create all three action buttons: download, check, cleanup (all as action buttons)
    buttons = [
        {
            'button_id': 'download_button',  # Use button_id instead of id
            'text': BUTTON_CONFIG['download']['text'],
            'style': 'success',  # Use success style for download
            'tooltip': BUTTON_CONFIG['download']['tooltip'],
            'order': 1
        },
        {
            'button_id': 'check_button',  # Use button_id instead of id
            'text': BUTTON_CONFIG['check']['text'],
            'style': BUTTON_CONFIG['check']['style'],
            'tooltip': BUTTON_CONFIG['check']['tooltip'],
            'order': 2
        },
        {
            'button_id': 'cleanup_button',  # Use button_id instead of id
            'text': BUTTON_CONFIG['cleanup']['text'], 
            'style': BUTTON_CONFIG['cleanup']['style'],
            'tooltip': BUTTON_CONFIG['cleanup']['tooltip'],
            'order': 3
        }
    ]
    
    
    # Create action container
    action_container = create_action_container(
        title="📥 Dataset Operations",
        buttons=buttons,
        show_save_reset=True
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
    
    # Extract buttons from action container
    action_buttons = action_container.get('buttons', {})
    
    # Ensure all required buttons exist and are properly referenced
    button_components = {}
    for btn_id in ['download_button', 'check_button', 'cleanup_button']:
        if btn_id in action_buttons and action_buttons[btn_id] is not None:
            button_components[btn_id] = action_buttons[btn_id]
        else:
            button_components[btn_id] = None
    
    # 6. Create components dictionary with all UI elements
    components = {
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,  # Full action container with all methods
        'operation_container': operation_container,
        'footer_container': footer_container,
        'ui_initialized': True,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        # Add direct button references
        **button_components
    }
    
    # 7. Create Main Container with all components
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
    
    # 8. Create the final UI components dictionary with proper null checks
    
    # Get operation container components directly using the correct names
    progress_tracker = operation_container.get('progress_tracker') if operation_container else None
    log_accordion = operation_container.get('log_accordion') if operation_container else None
    update_progress = operation_container.get('update_progress') if operation_container else None
    show_dialog = operation_container.get('show_dialog') if operation_container else None
    
    # For backward compatibility, set the old variable names
    progress_bar = progress_tracker
    log_output = log_accordion
    
    # Get status text from progress tracker if available
    status_text = None
    if hasattr(progress_tracker, 'get_status_text'):
        status_text = progress_tracker.get_status_text()
    
    
    
    # Create the final components dictionary
    components = {
        'ui': main_container,
        'ui_components': {
            'header': header_container,
            'form': form_container,
            'action': action_container,
            'operation': operation_container,
            'footer': footer_container
        },
        'ui_initialized': True,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        'config': config,
        'main_container': main_container,
        
        # Add container references
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        
        # Add all button references
        'download_button': action_buttons.get('download_button'),
        'check_button': action_buttons.get('check_button'),
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Add form widgets - accessed as attributes of input_options
        'workspace_input': input_options.workspace_input,
        'project_input': input_options.project_input,
        'version_input': input_options.version_input,
        'api_key_input': input_options.api_key_input,
        'validate_checkbox': input_options.validate_checkbox,
        'backup_checkbox': input_options.backup_checkbox,
        
        # Add operation widgets with the same instances as above
        'progress_bar': progress_bar,
        'status_text': status_text,
        'log_output': log_output,
        'update_progress': update_progress,
        'show_dialog': show_dialog
    }
    
    # Save/Reset functionality is now handled by the UIModule pattern
    # DownloaderUIModule provides: _extract_ui_config(), _update_status(), reset_downloader()
    # This eliminates the need for global_ui_handler
    
    return components








