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
    # Create all three action buttons: download, check, cleanup (all as action buttons)
    buttons = [
        {
            'id': 'download_btn',
            'text': BUTTON_CONFIG['download']['text'],
            'style': 'success',  # Use success style for download
            'tooltip': BUTTON_CONFIG['download']['tooltip'],
            'order': 1
        },
        {
            'id': 'check_btn', 
            'text': BUTTON_CONFIG['check']['text'],
            'style': BUTTON_CONFIG['check']['style'],
            'tooltip': BUTTON_CONFIG['check']['tooltip'],
            'order': 2
        },
        {
            'id': 'cleanup_btn',
            'text': BUTTON_CONFIG['cleanup']['text'], 
            'style': BUTTON_CONFIG['cleanup']['style'],
            'tooltip': BUTTON_CONFIG['cleanup']['tooltip'],
            'order': 3
        }
    ]
    
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
    
    # Create the UI components dictionary following the standard structure
    ui_components = {
        # Main containers
        'main_container': main_container,
        'header_container': header_container.container,
        'form_container': form_container['container'],
        'action_container': action_container['container'],
        'operation_container': operation_container['container'],
        'footer_container': footer_container.container,
        
        # Form widgets
        'form_widgets': {
            'workspace_input': getattr(input_options, 'workspace_input', None),
            'project_input': getattr(input_options, 'project_input', None),
            'version_input': getattr(input_options, 'version_input', None),
            'api_key_input': getattr(input_options, 'api_key_input', None),
            'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
            'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        },
        
        # Buttons - ensure we get the correct button references
        'buttons': {
            'download': action_container.get('buttons', {}).get('download_btn'),
            'check': action_container.get('buttons', {}).get('check_btn'), 
            'cleanup': action_container.get('buttons', {}).get('cleanup_btn'),
            'save': getattr(action_container.get('action_container'), 'save_button', None),
            'reset': getattr(action_container.get('action_container'), 'reset_button', None),
        },
        
        # Operation functions
        'operations': {
            'log_message': operation_container['log_message'],
            'update_progress': operation_container['update_progress'],
            'show_dialog': operation_container['show_dialog'],
        },
        
        # Module info
        'module_info': {
            'name': UI_CONFIG['module_name'],
            'parent': UI_CONFIG['parent_module'],
            'version': UI_CONFIG['version'],
            'initialized': True
        }
    }
    
    # Store the main container as 'ui' for backward compatibility
    components = {
        'ui': main_container,
        'ui_components': ui_components,
        'ui_initialized': True,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        'config': config.copy()
    }
    
    return components

def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    from .input_options import create_downloader_input_options
    
    # Create input options
    input_options = create_downloader_input_options(config)
    
    return {
        'container': input_options,
        'widgets': {
            'workspace_input': getattr(input_options, 'workspace_input', None),
            'project_input': getattr(input_options, 'project_input', None),
            'version_input': getattr(input_options, 'version_input', None),
            'api_key_input': getattr(input_options, 'api_key_input', None),
            'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
            'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> widgets.Widget:
    """
    Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Widget containing the summary content
    """
    summary = widgets.HTML(
        value="<h4>Dataset Download Summary</h4>"
             "<p>Configure your dataset download settings and click 'Download' to begin.</p>"
             "<ul>"
             f"<li>Workspace: {config.get('workspace', 'Not set')}</li>"
             f"<li>Project: {config.get('project', 'Not set')}</li>"
             f"<li>Version: {config.get('version', 'Not set')}</li>"
             "</ul>",
        layout=widgets.Layout(margin='10px 0')
    )
    return summary


def _create_module_info_box() -> widgets.Widget:
    """
    Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    from smartcash.ui.info_boxes.download_info import get_download_info
    return get_download_info(open_by_default=False)


def _create_module_tips_box() -> widgets.Widget:
    """
    Create the tips box content for the footer.
    
    Returns:
        Widget containing the tips content
    """
    tips = [
        "💡 Pastikan koneksi internet stabil saat mendownload dataset",
        "🔍 Selalu periksa status dataset sebelum mendownload",
        "⚠️ Simpan API key Anda dengan aman dan jangan membagikannya",
        "🔄 Gunakan tombol 'Check' untuk memverifikasi dataset sebelum download"
    ]
    
    return widgets.VBox([
        widgets.HTML("<h4>Tips & Best Practices</h4>"),
        widgets.VBox([
            widgets.HTML(f"<div style='margin: 5px 0;'>{tip}</div>")
            for tip in tips
        ])
    ])

