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
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

# Import downloader specific components
from .input_options import create_downloader_input_options


def create_downloader_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create downloader UI components following the colab/dependency pattern.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components and their references
    """
    if config is None:
        config = {}
    
    # Container for all child components following the established pattern
    child_components = {}

    # 1. Create Header Container
    header_container = create_header_container(
        title="📥 Dataset Downloader",
        subtitle="Download dataset Roboflow untuk SmartCash dengan UUID renaming dan validasi otomatis"
    )
    child_components['header_container'] = header_container.container
    
    # 2. Create Operation Container (centralized progress and logging)
    operation_container = create_operation_container(
        component_name="downloader_operation_container",
        show_progress=True,
        show_logs=True,
        initial_message="Dataset downloader ready...",
        log_height="200px"
    )
    child_components['operation_container'] = operation_container['container']
    child_components['operation_manager'] = operation_container  # For handlers to access
    
    # 3. Create downloader input form with improved design
    input_options = create_downloader_input_options(config)
    child_components['input_options'] = input_options
    
    # 4. Create Form Container to hold the input options
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    form_container['container'].children = (input_options,)
    child_components['form_container'] = form_container['container']
    
    # 5. Create Save/Reset buttons for configurations
    try:
        save_reset_components = create_save_reset_buttons(
            save_label="💾 Save Config",
            reset_label="🔄 Reset",
            with_sync_info=True
        )
        child_components['save_reset_buttons'] = save_reset_components
        child_components['save_button'] = save_reset_components.get('save_button')
        child_components['reset_button'] = save_reset_components.get('reset_button')
    except Exception:
        # Fallback if save/reset buttons fail
        child_components['save_reset_buttons'] = None
        child_components['save_button'] = None
        child_components['reset_button'] = None
    
    # 6. Create Action Container with specific downloader buttons
    action_container = create_action_container(
        buttons=[
            {
                'button_id': 'download_button',
                'text': '📥 Download',
                'style': 'primary',
                'icon': 'download',
                'tooltip': 'Download dataset from Roboflow',
                'order': 1
            },
            {
                'button_id': 'check_button', 
                'text': '🔍 Check Dataset',
                'style': 'info',
                'icon': 'search',
                'tooltip': 'Check dataset status and integrity',
                'order': 2
            },
            {
                'button_id': 'cleanup_button',
                'text': '🗑️ Cleanup Dataset',
                'style': 'danger',
                'icon': 'trash',
                'tooltip': 'Remove dataset files from local storage',
                'order': 3
            }
        ],
        title="🚀 Dataset Operations", 
        alignment="left"
    )
    child_components['action_container'] = action_container['container']
    child_components['download_button'] = action_container['buttons'].get('download_button')
    child_components['check_button'] = action_container['buttons'].get('check_button')
    child_components['cleanup_button'] = action_container['buttons'].get('cleanup_button')
    child_components['action_container_manager'] = action_container  # For button management
    
    # 7. Create Footer Container with enhanced tips
    import ipywidgets as widgets
    footer_container = create_footer_container(
        info_box=widgets.HTML(
            value="""
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>💡 Dataset Download Tips:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>API Key akan otomatis terdeteksi dari Colab Secrets (ROBOFLOW_API_KEY)</li>
                    <li>UUID renaming akan mengubah nama file untuk konsistensi</li>
                    <li>Validasi memverifikasi integritas dataset setelah download</li>
                    <li>Backup melindungi data existing sebelum replace</li>
                    <li>Format output: YOLOv5 PyTorch dengan struktur train/valid/test</li>
                </ul>
            </div>
            """
        )
    )
    child_components['footer_container'] = footer_container.container
    
    # 8. Create Operation Summary for displaying results
    from .operation_summary import create_operation_summary
    operation_summary = create_operation_summary("🔧 Dataset downloader ready...")
    child_components['operation_summary'] = operation_summary
    
    # 9. Create Main Container using latest pattern
    main_container = create_main_container(
        # Define the layout structure with explicit ordering
        components=[
            # Header section
            {
                'type': 'header',
                'component': child_components['header_container'],
                'order': 0,
                'name': 'header'
            },
            # Operation container (progress and logs)
            {
                'type': 'operation',
                'component': child_components['operation_container'],
                'order': 1,
                'name': 'operations'
            },
            # Form section (input options)
            {
                'type': 'form',
                'component': child_components['form_container'],
                'order': 2,
                'name': 'form'
            },
            # Config buttons (if available)
            {
                'type': 'action',
                'component': child_components.get('save_reset_buttons', {}).get('container'),
                'order': 3,
                'name': 'config_buttons'
            },
            # Main action container
            {
                'type': 'action',
                'component': child_components['action_container'],
                'order': 4,
                'name': 'actions'
            },
            # Operation summary for results
            {
                'type': 'summary',
                'component': child_components['operation_summary'],
                'order': 5,
                'name': 'operation_summary'
            },
            # Footer
            {
                'type': 'footer',
                'component': child_components['footer_container'],
                'order': 6,
                'name': 'footer'
            }
        ],
        # Styling options
        container_style={
            'width': '100%',
            'padding': '20px',
            'border': '1px solid #ddd',
            'border_radius': '10px'
        }
    )
    
    # Store the main container and its UI reference
    child_components['main_container'] = main_container.container
    child_components['ui'] = main_container.container

    # Add input component references for easy access
    child_components.update({
        'workspace_input': getattr(input_options, 'workspace_input', None),
        'project_input': getattr(input_options, 'project_input', None),
        'version_input': getattr(input_options, 'version_input', None),
        'api_key_input': getattr(input_options, 'api_key_input', None),
        'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
    })
    
    return child_components