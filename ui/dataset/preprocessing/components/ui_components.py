"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Assembly komponen UI utama untuk preprocessing
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
from .advanced_options import create_preprocessing_advanced_options


def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Buat komponen UI utama untuk preprocessing dengan integrasi baru.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        Dict[str, Any]: Dictionary komponen UI
    """
    # Header
    header = create_header(
        f"{ICONS['processing']} Dataset Preprocessing", 
        "Preprocessing dataset untuk training model SmartCash dengan integrasi Drive"
    )
    
    # Status panel
    status_panel = create_status_panel(
        "Siap memulai preprocessing dataset", "info"
    )
    
    # Input options (kolom kiri)
    input_options = create_preprocessing_input_options(config)
    
    # Advanced options (kolom kanan)
    advanced_options = create_preprocessing_advanced_options(config)
    
    # Layout 2 kolom
    options_container = widgets.HBox([
        input_options,
        advanced_options
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Action buttons dengan tombol stop
    action_buttons = create_action_buttons(
        primary_label="Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[
            ("Stop", "stop", "warning")
        ],
        cleanup_enabled=True
    )
    
    # Hide stop button initially
    if 'stop_button' in action_buttons:
        action_buttons['stop_button'].layout.display = 'none'
    
    # Save & reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan Konfigurasi",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi ke file dan sinkronkan dengan Drive",
        reset_tooltip="Reset konfigurasi ke nilai default"
    )
    
    # Sync info message
    sync_info = create_sync_info_message(
        message="Konfigurasi dan data preprocessing akan otomatis disinkronkan dengan Google Drive.",
        icon="cloud",
        color=COLORS['info']
    )
    
    # Confirmation area untuk dialog
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            margin='15px 0',
            min_height='30px',
            display='none',  # Hidden by default
            border='1px solid #ddd',
            padding='10px',
            border_radius='5px'
        )
    )
    
    # Progress tracking dengan 2-level
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
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Konfigurasi Preprocessing</h4>"),
        options_container,
        widgets.VBox([
            save_reset_buttons['container'],
            sync_info['container']
        ], layout=widgets.Layout(align_items='flex-end', width='100%')),
        create_divider(),
        action_buttons['container'],
        confirmation_area,
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
        'advanced_options': advanced_options,
        'options_container': options_container,
        
        # Individual input widgets (untuk backward compatibility)
        'preprocess_options': input_options,  # Alias
        'resolution_dropdown': input_options.resolution_dropdown,
        'normalization_dropdown': input_options.normalization_dropdown,
        'preserve_aspect_ratio_checkbox': input_options.preserve_aspect_ratio_checkbox,
        'augmentation_checkbox': input_options.augmentation_checkbox,
        'force_reprocess_checkbox': input_options.force_reprocess_checkbox,
        'worker_slider': advanced_options.worker_slider,
        'split_selector': advanced_options.split_selector,
        'reverse_split_map': advanced_options.reverse_split_map,
        
        # Action buttons
        'action_buttons': action_buttons,
        'preprocess_button': action_buttons['download_button'],  # Primary button
        'preprocessing_button': action_buttons['download_button'],  # Alias
        'stop_button': action_buttons.get('secondary_buttons', [{}])[0] if action_buttons.get('secondary_buttons') else None,
        'cleanup_button': action_buttons['cleanup_button'],
        
        # Save/reset buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # UI areas
        'confirmation_area': confirmation_area,
        'sync_info': sync_info,
        
        # Progress components
        'progress_components': progress_components,
        'progress_container': progress_components['progress_container'],
        'progress_bar': progress_components['progress_bar'],
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label'),
        'current_progress': progress_components.get('current_progress'),
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],  # Alias untuk backward compatibility
        
        # Module info
        'module_name': 'preprocessing',
        'help_panel': help_panel,
        
        # Default paths
        'data_dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    }
    
    return ui_components