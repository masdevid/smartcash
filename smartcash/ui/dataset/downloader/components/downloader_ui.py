"""Dataset Downloader UI components following SmartCash UI standards.

Implements a container-based UI with form inputs, action buttons, and progress tracking.
"""

from typing import Dict, Any, Optional

# Standard container imports
from smartcash.ui.components import (create_header_container, create_form_container, create_action_container, 
                                  create_operation_container, create_summary_container, create_main_container)
from smartcash.ui.components.form_container import LayoutType
from smartcash.ui.core.decorators import handle_ui_errors
from smartcash.ui.dataset.downloader.constants import (
    UI_CONFIG, BUTTON_CONFIG, VALIDATION_RULES,
    OperationType, UIComponent, ButtonStyle
)
from .input_options import create_downloader_input_options

# Re-export constants for backward compatibility
__all__ = ['UI_CONFIG', 'BUTTON_CONFIG', 'VALIDATION_RULES', 'create_downloader_ui_components']


@handle_ui_errors(error_component_title="Dataset Downloader UI Creation Error")
def create_downloader_ui_components(
    config: Optional[Dict[str, Any]] = None, 
    **kwargs
) -> Dict[str, Any]:
    """Create and initialize the Dataset Downloader UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments including module_config
        
    Returns:
        Dict: UI components and references
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
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        'config': config.copy()
    }
    
    # 1. Create header with title and subtitle
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='📥',  # Download emoji for downloader
        show_environment=True,
        environment='local',  # Default environment
        config_path='dataset_config.yaml'
    )
    components['header_container'] = header_container
    
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
    # Create action buttons using OperationType enum
    buttons = [
        {
            'button_id': UIComponent.button_id(OperationType.DOWNLOAD),
            'text': BUTTON_CONFIG[OperationType.DOWNLOAD.value]['text'],
            'style': BUTTON_CONFIG[OperationType.DOWNLOAD.value]['style'],
            'tooltip': BUTTON_CONFIG[OperationType.DOWNLOAD.value]['tooltip'],
            'order': 1
        },
        {
            'button_id': UIComponent.button_id(OperationType.CHECK),
            'text': BUTTON_CONFIG[OperationType.CHECK.value]['text'],
            'style': BUTTON_CONFIG[OperationType.CHECK.value]['style'],
            'tooltip': BUTTON_CONFIG[OperationType.CHECK.value]['tooltip'],
            'order': 2
        },
        {
            'button_id': UIComponent.button_id(OperationType.CLEANUP),
            'text': BUTTON_CONFIG[OperationType.CLEANUP.value]['text'],
            'style': BUTTON_CONFIG[OperationType.CLEANUP.value]['style'],
            'tooltip': BUTTON_CONFIG[OperationType.CLEANUP.value]['tooltip'],
            'order': 3
        }
    ]
    
    
    # Create action container with save/reset buttons as requested
    action_container = create_action_container(
        title="📥 Operasi Dataset",
        buttons=buttons,
        show_save_reset=True
    )
    
    # 4. Create Operation Container with dual progress tracker for download operations
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name=UI_CONFIG['module_name'],
        log_height="150px",
        collapsible=True,
        collapsed=False
    )
    
    # Get progress tracker and log accordion from operation container
    progress_tracker = operation_container.get('progress_tracker')
    log_accordion = operation_container.get('log_accordion')
    
    # 5. Create Summary Container for operation results
    summary_container = create_summary_container(
        theme='default',
        title='Ringkasan Unduhan',
        icon='📊'
    )
    summary_content = "<div style='padding: 10px; width: 100%;'>Ringkasan operasi unduhan akan muncul di sini...</div>"
    summary_container.set_content(summary_content)
    
    # Initialize progress tracker if available
    if progress_tracker and hasattr(progress_tracker, 'initialize'):
        progress_tracker.initialize()
        # Start with the first step if the progress tracker supports it
        if hasattr(progress_tracker, 'start'):
            progress_tracker.start()
    
    
    # Extract and validate action buttons with safe access
    action_buttons = action_container.get('buttons', {})
    
    # Map operation types to their button references with null checks
    button_components = {}
    for op in OperationType:
        btn_id = UIComponent.button_id(op)
        button_ref = action_buttons.get(btn_id)
        if button_ref is not None:
            button_components[btn_id] = button_ref
    
    # 6. Create components dictionary with all UI elements
    components = {
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,  # Full action container with all methods
        'summary_container': summary_container,  # Operation summary container
        'operation_container': operation_container,
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
            {'component': summary_container.container, 'type': 'summary'},
            {'component': operation_container['container'], 'type': 'operation'}
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
    
    # Try to get progress tracker from widget if not found directly
    if not progress_tracker and hasattr(operation_container, 'widget'):
        progress_tracker = getattr(operation_container.widget, 'progress_tracker', None)
    
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
            'operation': operation_container
        },
        'ui_initialized': True,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        'config': config,
        'main_container': main_container.container,
        
        # Add container references
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'summary_container': summary_container,
        'operation_container': operation_container,
        
        # Add all button references with safe access and consistent naming
        'download': action_buttons.get('download') if action_buttons else None,
        'check': action_buttons.get('check') if action_buttons else None,
        'cleanup': action_buttons.get('cleanup') if action_buttons else None,
        
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
        'log_accordion': log_accordion,  # Use consistent naming
        'update_progress': update_progress,
        'show_dialog': show_dialog
    }
    
    return components