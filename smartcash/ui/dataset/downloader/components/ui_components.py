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
from typing import Dict, Any

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.footer_container import create_footer_container

# Import other UI components
from smartcash.ui.components import create_log_accordion
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.action_container import create_action_container

# Import downloader specific components
from .input_options import create_downloader_input_options


def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create and configure the main downloader UI interface.
    
    This function orchestrates the creation of all UI components for the dataset
    downloader, including headers, input forms, progress tracking, and action buttons.
    It uses shared container components to maintain consistency with the rest of
    the SmartCash UI.
    
    Args:
        config: Optional configuration dictionary containing:
            - downloader: Downloader-specific configuration
            - ui: UI-specific settings
            - Any other configuration needed by child components
            
    Returns:
        Dictionary containing all UI components with the following structure:
        {
            'ui': widgets.Widget - The main UI container widget
            'header_container': widgets.Widget - Header section
            'input_options': dict - Input form components
            'progress_tracker': ProgressTracker - Progress tracking component
            'log_output': widgets.Output - Log display area
            'action_buttons': dict - Action button widgets
            'status_panel': widgets.Widget - Status display area
        }
        
    Raises:
        ValueError: If required configuration is missing or invalid
        RuntimeError: If UI component creation fails
    """
    # Initialize ui_components dictionary to store all UI components
    ui_components = {}

    # 1. Create Header Container
    header_container = create_header_container(
        title="📥 Dataset Downloader", 
        subtitle="Download dataset Roboflow untuk SmartCash training dengan UUID renaming otomatis"
    )
    ui_components['header_container'] = header_container
    
    # 2. Create Form Container with input options and save/reset buttons
    # Input options
    input_options = create_downloader_input_options(config)
    ui_components['input_options'] = input_options
    
    # Create form container with save/reset buttons
    form_components = create_form_container()
    
    # Add input options to the form container
    form_components['form_container'].children = (input_options,)
    
    # Store form components
    ui_components['form_container'] = form_components['container']
    ui_components['save_button'] = form_components['save_button']
    ui_components['reset_button'] = form_components['reset_button']
    
    # 3. Create Summary Container for help content
    help_content = """
    <div>
        <p>Download dataset dari Roboflow dengan UUID renaming dan validasi otomatis.</p>
        <div>
            <strong>Parameter Utama:</strong>
            <ul>
                <li><strong>Workspace/Project:</strong> Identifikasi dataset Roboflow</li>
                <li><strong>Version:</strong> Versi dataset yang akan didownload</li>
                <li><strong>API Key:</strong> Auto-detect dari Colab secrets</li>
                <li><strong>Validasi:</strong> Verifikasi integritas dataset</li>
                <li><strong>Backup:</strong> Backup data sebelum replace</li>
            </ul>
        </div>
        <div>
            <strong>Fitur Utama:</strong>
            <ul>
                <li><strong>UUID Renaming:</strong> Rename semua file dengan UUID</li>
                <li><strong>Validasi:</strong> Verifikasi integritas dataset</li>
                <li><strong>Backup:</strong> Backup data sebelum replace</li>
                <li><strong>Format:</strong> YOLOv5 PyTorch</li>
            </ul>
        </div>
    </div>
    """
    
    summary_container = create_summary_container(
        theme="info",
        title="Dataset Information",
        icon="📋"
    )
    summary_container.set_content(help_content)
    ui_components['summary_container'] = summary_container
    
    # 4. Create Action Container
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "download",
                "text": "📥 Download Dataset",
                "style": "primary",
                "order": 1
            },
            {
                "button_id": "check",
                "text": "🔍 Check Dataset",
                "style": "info",
                "order": 2
            },
            {
                "button_id": "cleanup",
                "text": "🗑️ Bersihkan Dataset",
                "style": "warning",
                "tooltip": "Hapus dataset yang sudah didownload",
                "order": 3
            }
        ],
        title="🚀 Dataset Operations",
        alignment="left"
    )
    
    # Get buttons from the action container
    download_button = action_container['buttons']['download']
    check_button = action_container['buttons']['check']
    cleanup_button = action_container['buttons']['cleanup']
    
    # Store action container in ui_components
    ui_components.update({
        'action_container': action_container,
        'download_button': download_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'confirmation_area': action_container['dialog_area'],
        'buttons': [download_button, check_button, cleanup_button],
        'show_dialog': action_container['show_dialog'],
        'show_info': action_container['show_info'],
        'clear_dialog': action_container['clear_dialog'],
        'is_dialog_visible': action_container['is_dialog_visible']
    })
    
    # Use the action container as the action section
    action_section = action_container['container']
    ui_components['action_section'] = action_section
    
    # 6. Create Progress Tracker
    progress_tracker = ProgressTracker()
    progress_tracker.show()
    ui_components['progress_tracker'] = progress_tracker
    
    # 7. Create Log Accordion
    log_accordion = create_log_accordion()
    ui_components['log_accordion'] = log_accordion
    ui_components['log_output'] = log_accordion  # Alias for compatibility
    ui_components['log_components'] = log_accordion
    
    # 8. Create Footer Container
    footer_container = create_footer_container()
    ui_components['footer_container'] = footer_container
    
    # 9. Assemble Main Container
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=form_components['container'],
        summary_container=summary_container.container,
        action_container=action_section
    )
    
    # Store the main UI container
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container
    
    # Add additional references for backward compatibility
    ui_components.update({
        'status_panel': header_container.status_panel,
        'module_name': 'downloader',
        
        # Input components (from input_options)
        'workspace_input': getattr(input_options, 'workspace_input', None),
        'project_input': getattr(input_options, 'project_input', None), 
        'version_input': getattr(input_options, 'version_input', None),
        'api_key_input': getattr(input_options, 'api_key_input', None),
        'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        
        # Progress components
        'progress_container': progress_tracker.container if hasattr(progress_tracker, 'container') else None,
        'show_for_operation': getattr(progress_tracker, 'show', None),
        'update_progress': getattr(progress_tracker, 'update', None),
        'complete_operation': getattr(progress_tracker, 'complete', None),
        'error_operation': getattr(progress_tracker, 'error', None),
        'reset_all': getattr(progress_tracker, 'reset', None),
        
        # Status for compatibility
        'status': log_accordion.get('log_output')
    })
    
    return ui_components