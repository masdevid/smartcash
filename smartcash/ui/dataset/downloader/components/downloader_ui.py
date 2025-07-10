"""
Dataset Downloader UI Components Module.

This module provides UI components for the dataset downloader interface, built using
shared container components from the SmartCash UI library. It implements the visual
representation and layout of the dataset downloader functionality.

Components:
- Main downloader UI container
- Input form elements
- Progress tracking
- Log display
- Action buttons

The module follows a component-based architecture, where each major UI section is
created as a separate component and then composed together to form the complete UI.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import standard container components following colab/dependency pattern
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.action_container import create_action_container

# Import downloader specific components
from .input_options import create_downloader_input_options


def create_downloader_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create downloader UI components following the standard layout template.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components and their references
    """
    if config is None:
        config = {}
    
    # Container for all child components
    components = {}

    # 1. Create Header Container
    header = create_header_container(
        title="📥 Dataset Downloader",
        subtitle="Download dataset Roboflow untuk SmartCash dengan UUID renaming dan validasi otomatis",
        status_text="Ready"
    )
    components['header'] = header
    
    # 2. Create Form Container
    form = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding="0", # no need padding to span the same width
        gap="12px"
    )
    
    # Add input options to form
    input_options = create_downloader_input_options(config)
    form['add_item'](input_options, "input_options")
    components['form'] = form
    
    # 3. Create Action Buttons with primary button for main download action
    action_buttons = create_action_container(
        title="Download Actions",
        buttons=[
            {
                'id': 'check', 
                'text': '🔍 Check',
                'style': 'info',
                'tooltip': 'Check dataset status and integrity',
                'order': 1
            },
            {
                'id': 'cleanup',
                'text': '🗑️ Cleanup',
                'style': 'danger',
                'tooltip': 'Remove dataset files from local storage',
                'order': 2
            }
        ],
        alignment="left",
        show_save_reset=True  # Use default save/reset buttons
    )
    
    # Configure the primary button for main download action
    primary_button = action_buttons['primary_button']
    if primary_button:
        primary_button.description = "📥 Download"
        primary_button.tooltip = "Download dataset from Roboflow"
    
    components['actions'] = action_buttons
    components['download_button'] = primary_button
    components['check_button'] = action_buttons['buttons'].get('check')
    components['cleanup_button'] = action_buttons['buttons'].get('cleanup')
    components['save_reset'] = action_buttons['action_container']
    # 4. Create Operation Container
    operation = create_operation_container(
        show_progress=True,
        show_logs=True,
        log_module_name="Dataset Downloader",
        log_height="200px"
    )
    components['operation'] = operation
    
    # 5. Create Footer Container with Download Info
    from smartcash.ui.info_boxes.download_info import get_download_info
    
    # Get the download info accordion
    download_info = get_download_info(open_by_default=False)
    
    # Create a container for the footer
    footer = widgets.VBox(
        [download_info],
        layout=widgets.Layout(
            border_top='1px solid #dee2e6',
            background='#f8f9fa',
            width='100%',
            margin='20px 0'
        )
    )
    components['footer'] = footer
    
    # 6. Create Main Container
    main_container = create_main_container(
        components=[
            {'component': header.container, 'type': 'header'},
            {'component': form['container'], 'type': 'form'},
            {'component': action_buttons['container'], 'type': 'action'},
            {'component': operation['container'], 'type': 'operation'},
            {'component': footer, 'type': 'footer'}
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
    components['container'] = main_container
    components['ui'] = main_container
    
    # Add input component references for easy access
    components.update({
        'workspace_input': getattr(input_options, 'workspace_input', None),
        'project_input': getattr(input_options, 'project_input', None),
        'version_input': getattr(input_options, 'version_input', None),
        'api_key_input': getattr(input_options, 'api_key_input', None),
        'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
    })
    
    # Add button references
    buttons = action_buttons['buttons']
    components.update({
        'download_button': buttons.get('download'),
        'check_button': buttons.get('check'),
        'cleanup_button': buttons.get('cleanup')
    })

    return components