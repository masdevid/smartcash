"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Assembly komponen UI yang disederhanakan untuk preprocessing
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info

# Import shared components
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.sync_info_message import create_sync_info_message

# Import local components
from .input_options import create_preprocessing_input_options


def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Buat komponen UI yang disederhanakan untuk preprocessing.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        Dict[str, Any]: Dictionary komponen UI
    """
    # Header
    header = create_header(
        f"{ICONS['processing']} Dataset Preprocessing", 
        "Preprocessing dataset untuk training model SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel(
        "Siap memulai preprocessing dataset", "info"
    )
    
    # Input options (simplified)
    input_options = create_preprocessing_input_options(config)
    
    # Action buttons dengan cleanup
    action_buttons = create_action_buttons(
        primary_label="Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[],
        cleanup_enabled=True
    )
    
    # Save & reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan Config",
        reset_label="Reset"
    )
    
    # Sync info message
    sync_info = create_sync_info_message(
        message="Konfigurasi akan otomatis disinkronkan dengan Google Drive.",
        icon="cloud"
    )
    
    # Progress tracking
    progress_components = create_progress_tracking(
        module_name='preprocessing',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )
    
    # Log accordion
    log_components = create_log_accordion(
        module_name='preprocessing',
        height='250px',
        width='100%'
    )
    
    # Help info
    help_panel = get_preprocessing_info()
    
    # Main UI assembly
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Konfigurasi</h4>"),
        input_options,
        widgets.VBox([
            save_reset_buttons['container'],
            sync_info['container']
        ], layout=widgets.Layout(align_items='flex-end', width='100%')),
        create_divider(),
        action_buttons['container'],
        progress_components['progress_container'],
        log_components['log_accordion'],
        help_panel
    ])
    
    # Kompile semua komponen
    ui_components = {
        # Main UI
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        
        # Input components
        'input_options': input_options,
        'resolution_dropdown': input_options.resolution_dropdown,
        'normalization_dropdown': input_options.normalization_dropdown, 
        'worker_slider': input_options.worker_slider,
        'split_dropdown': input_options.split_dropdown,
        
        # Action buttons
        'action_buttons': action_buttons,
        'preprocess_button': action_buttons['download_button'],  # Primary button
        'cleanup_button': action_buttons['cleanup_button'],
        
        # Save/reset buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # UI areas
        'sync_info': sync_info,
        
        # Progress components
        'progress_components': progress_components,
        'progress_container': progress_components['progress_container'],
        'progress_bar': progress_components['progress_bar'],
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label'),
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],  # Alias
        
        # Module info
        'module_name': 'preprocessing',
        'help_panel': help_panel,
        
        # Default paths
        'data_dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    }
    
    return ui_components