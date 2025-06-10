"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Fixed UI components dengan confirmation area yang benar, dual progress warna seragam, dan layout yang optimal
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fixed preprocessing UI dengan confirmation area yang benar dan layout optimal"""
    
    config = config or {}
    
    # === CORE COMPONENTS ===
    header = create_header("🔧 Dataset Preprocessing", "Preprocessing dataset dengan validasi dan real-time progress")
    status_panel = create_status_panel("🚀 Siap memulai preprocessing dataset", "info")
    input_options = create_preprocessing_input_options(config)
    
    # === SAVE/RESET BUTTONS ===
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        with_sync_info=True, sync_message="Konfigurasi disinkronkan dengan backend"
    )
    
    # === ACTION BUTTONS ===
    action_buttons = create_action_buttons(
        primary_label="🚀 Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[("🔍 Check Dataset", "search", "info")],
        cleanup_enabled=True,
        button_width='180px'
    )
    
    # === CONFIRMATION AREA (Fixed) ===
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', 
        min_height='50px',
        max_height='200px', 
        margin='10px 0',
        padding='5px',
        border='1px solid #e0e0e0',
        border_radius='4px',
        overflow='auto',
        background_color='#fafafa'
    ))
    
    # === DUAL PROGRESS TRACKER (Shared Component) ===
    progress_tracker_components = create_dual_progress_tracker(
        operation="Dataset Preprocessing",
        auto_hide=False
    )
    
    # Extract progress tracker dari components
    if isinstance(progress_tracker_components, dict):
        progress_tracker = progress_tracker_components.get('tracker')
        progress_container = progress_tracker_components.get('container', widgets.VBox([]))
    else:
        progress_tracker = progress_tracker_components
        progress_container = widgets.VBox([progress_tracker])
    
    # === LOG ACCORDION (Fixed) ===
    log_components = create_log_accordion(module_name='preprocessing', height='250px')
    log_output = log_components['log_output']
    log_accordion = log_components['log_accordion']
    
    # === ACTION SECTION (Fixed layout seperti augmentation) ===
    action_section = widgets.VBox([
        _create_section_header("🚀 Pipeline Operations", "#28a745"),
        action_buttons['container'],
        widgets.HTML("<div style='margin: 5px 0 2px 0; font-size: 13px; color: #666;'><strong>📋 Status & Konfirmasi:</strong></div>"),
        confirmation_area
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        padding='12px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#f9f9f9'
    ))
    
    # === CONFIG SECTION ===
    config_section = widgets.VBox([
        widgets.Box([save_reset_buttons['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(margin='8px 0'))
    
    # === MAIN UI ASSEMBLY (Fixed layout) ===
    ui = widgets.VBox([
        header,
        status_panel,
        input_options,
        config_section,
        action_section,
        progress_container,
        log_accordion
    ], layout=widgets.Layout(
        width='100%', 
        max_width='100%',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        overflow='hidden'
    ))
    
    # === COMPONENTS MAPPING ===
    ui_components = {
        'ui': ui,
        
        # Action buttons
        'preprocess_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Save/reset
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Communication areas
        'confirmation_area': confirmation_area,
        'status_panel': status_panel,
        
        # Progress tracking
        'progress_tracker': progress_tracker,
        'progress_container': progress_container,
        
        # Log
        'log_output': log_output,
        'log_accordion': log_accordion,
        'status': log_output,
        
        # Input components
        'input_options': input_options,
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None),
        'target_splits_select': getattr(input_options, 'target_splits_select', None),
        'batch_size_input': getattr(input_options, 'batch_size_input', None),
        'validation_checkbox': getattr(input_options, 'validation_checkbox', None),
        'preserve_aspect_checkbox': getattr(input_options, 'preserve_aspect_checkbox', None),
        'move_invalid_checkbox': getattr(input_options, 'move_invalid_checkbox', None),
        'invalid_dir_input': getattr(input_options, 'invalid_dir_input', None),
        
        # Module metadata
        'module_name': 'preprocessing',
        'ui_initialized': True
    }
    
    return ui_components

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header"""
    return widgets.HTML(f"""
    <h4 style="color: #333; margin: 8px 0 6px 0; border-bottom: 2px solid {color}; 
               font-size: 14px; padding-bottom: 4px; font-weight: 600;">
        {title}
    </h4>
    """)