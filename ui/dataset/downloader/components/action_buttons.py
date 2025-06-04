"""
File: smartcash/ui/dataset/downloader/components/action_buttons.py
Deskripsi: Action buttons dengan save/reset integration dan responsive layout
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.layout_utils import create_responsive_container
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

def create_action_buttons() -> Dict[str, Any]:
    """Create action buttons dengan save/reset integration."""
    
    # Main action buttons
    main_buttons = _create_main_action_buttons()
    
    # Save/Reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="💾 Save Config",
        reset_label="🔄 Reset",
        save_tooltip="Simpan konfigurasi download saat ini",
        reset_tooltip="Reset ke konfigurasi default",
        with_sync_info=True,
        sync_message="Konfigurasi akan tersimpan di dataset_config.yaml",
        button_width="120px"
    )
    
    # Validation button
    validate_button = widgets.Button(
        description="🔍 Validate",
        button_style='info',
        tooltip='Validate parameter dan koneksi API',
        icon='check-circle',
        layout=widgets.Layout(width='120px', height='35px', margin='2px')
    )
    
    # Main download button (prominent)
    download_button = widgets.Button(
        description="📥 Download Dataset",
        button_style='success',
        tooltip='Mulai download dataset dari Roboflow',
        icon='download',
        layout=widgets.Layout(width='180px', height='40px', margin='5px')
    )
    
    # Secondary buttons row
    secondary_buttons = create_responsive_container([
        validate_button,
        save_reset_buttons['save_button'],
        save_reset_buttons['reset_button']
    ], container_type="hbox", justify_content="center", padding="5px")
    
    # Main button row
    main_button_row = create_responsive_container([
        download_button
    ], container_type="hbox", justify_content="center", padding="10px")
    
    # Combined container dengan spacing
    container = create_responsive_container([
        secondary_buttons,
        main_button_row,
        save_reset_buttons['sync_info'] if save_reset_buttons.get('sync_info') else widgets.HTML("")
    ], container_type="vbox", align_items="center", padding="10px")
    
    return {
        'container': container,
        'download_button': download_button,
        'validate_button': validate_button,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'main_buttons': main_buttons,
        'secondary_buttons': secondary_buttons,
        'save_reset_components': save_reset_buttons
    }

def _create_main_action_buttons() -> Dict[str, Any]:
    """Create main action buttons dengan consistent styling."""
    
    # Download button (primary action)
    download_button = widgets.Button(
        description="📥 Download Dataset",
        button_style='success',
        tooltip='Download dataset dari Roboflow dan organize ke struktur final',
        icon='download',
        layout=widgets.Layout(
            width='180px',
            height='40px',
            margin='5px'
        )
    )
    
    # Quick validate button
    quick_validate_button = widgets.Button(
        description="⚡ Quick Check",
        button_style='warning',
        tooltip='Quick validation tanpa download',
        icon='bolt',
        layout=widgets.Layout(
            width='120px',
            height='35px',
            margin='2px'
        )
    )
    
    return {
        'download_button': download_button,
        'quick_validate_button': quick_validate_button
    }

def create_status_action_bar() -> Dict[str, Any]:
    """Create status action bar untuk quick actions."""
    
    # Quick status indicators
    api_status = widgets.HTML(
        value="<span style='color:#666;'>🔑 API: Not checked</span>",
        layout=widgets.Layout(width='auto', margin='5px')
    )
    
    dataset_status = widgets.HTML(
        value="<span style='color:#666;'>📊 Dataset: Unknown</span>",
        layout=widgets.Layout(width='auto', margin='5px')
    )
    
    connection_status = widgets.HTML(
        value="<span style='color:#666;'>🌐 Connection: Unknown</span>",
        layout=widgets.Layout(width='auto', margin='5px')
    )
    
    # Quick action buttons
    refresh_status_button = widgets.Button(
        description="🔄",
        button_style='',
        tooltip='Refresh status indicators',
        layout=widgets.Layout(width='40px', height='30px', margin='2px')
    )
    
    clear_logs_button = widgets.Button(
        description="🗑️",
        button_style='',
        tooltip='Clear log output',
        layout=widgets.Layout(width='40px', height='30px', margin='2px')
    )
    
    # Status container
    status_container = create_responsive_container([
        api_status,
        dataset_status,
        connection_status
    ], container_type="hbox", justify_content="flex-start", padding="5px")
    
    # Actions container
    actions_container = create_responsive_container([
        refresh_status_button,
        clear_logs_button
    ], container_type="hbox", justify_content="flex-end", padding="5px")
    
    # Main container
    container = create_responsive_container([
        status_container,
        actions_container
    ], container_type="hbox", justify_content="space-between", padding="5px")
    
    # Add styling
    container.layout.border = '1px solid #e0e0e0'
    container.layout.border_radius = '4px'
    container.layout.background_color = '#fafafa'
    container.layout.margin = '5px 0'
    
    return {
        'container': container,
        'api_status': api_status,
        'dataset_status': dataset_status,
        'connection_status': connection_status,
        'refresh_status_button': refresh_status_button,
        'clear_logs_button': clear_logs_button
    }

def update_button_states(ui_components: Dict[str, Any], state: str) -> None:
    """Update button states berdasarkan current operation state."""
    states = {
        'ready': {
            'download_button': {'disabled': False, 'description': '📥 Download Dataset'},
            'validate_button': {'disabled': False, 'description': '🔍 Validate'},
            'save_button': {'disabled': False},
            'reset_button': {'disabled': False}
        },
        'downloading': {
            'download_button': {'disabled': True, 'description': '⏳ Downloading...'},
            'validate_button': {'disabled': True, 'description': '🔍 Validate'},
            'save_button': {'disabled': True},
            'reset_button': {'disabled': True}
        },
        'validating': {
            'download_button': {'disabled': True, 'description': '📥 Download Dataset'},
            'validate_button': {'disabled': True, 'description': '⏳ Validating...'},
            'save_button': {'disabled': False},
            'reset_button': {'disabled': False}
        },
        'error': {
            'download_button': {'disabled': False, 'description': '📥 Retry Download'},
            'validate_button': {'disabled': False, 'description': '🔍 Validate'},
            'save_button': {'disabled': False},
            'reset_button': {'disabled': False}
        }
    }
    
    current_states = states.get(state, states['ready'])
    
    for button_key, button_state in current_states.items():
        if button_key in ui_components and ui_components[button_key]:
            button = ui_components[button_key]
            for attr, value in button_state.items():
                if hasattr(button, attr):
                    setattr(button, attr, value)