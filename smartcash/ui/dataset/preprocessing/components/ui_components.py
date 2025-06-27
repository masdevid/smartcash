"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Complete UI components dengan proper initialization dan dialog integration
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create preprocessing UI dengan API integration dan dialog support"""
    config = config or {}
    
    # Initialize UI components dictionary first
    ui_components = {}
    
    # Import dengan error handling
    from smartcash.ui.components import (
        create_header, create_action_buttons, create_status_panel,
        create_log_accordion, create_save_reset_buttons,
        create_dual_progress_tracker, create_confirmation_area
    )

    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
    
    # === CORE COMPONENTS ===
    
    # Header
    header = create_header(
        "🔧 Dataset Preprocessing", 
        "Preprocessing dataset dengan YOLO normalization dan real-time progress",
        "🚀"
    )
    
    # Status panel
    status_panel = create_status_panel(
        "🚀 Siap memulai preprocessing dengan API baru", 
        "info"
    )
    
    # Input options
    input_options = create_preprocessing_input_options(config)
    
    # Save/Reset buttons
    save_reset_components = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset", 
        with_sync_info=False
    )
    
    # Action buttons
    action_components = create_action_buttons(
        primary_label="🚀 Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[("🔍 Check Dataset", "search", "info")],
        cleanup_enabled=True,
        button_width='180px'
    )
    
    # Progress tracker
    progress_tracker = create_dual_progress_tracker(
        operation="Dataset Preprocessing",
        auto_hide=False
    )
    
    # Get container from tracker
    progress_container = progress_tracker.container if hasattr(progress_tracker, 'container') else widgets.VBox([])
    
    # Log components
    log_components = create_log_accordion(
        module_name='preprocessing',
        height='200px'
    )
    
    # === DIALOG AREA ===
    
    # Confirmation area component
    confirmation_area, _ = create_confirmation_area()
    
    # === LAYOUT SECTIONS ===
    
    # Config section dengan save/reset buttons
    config_section = widgets.VBox([
        widgets.Box([save_reset_components['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(margin='8px 0'))
    
    # Action section dengan confirmation area
    action_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#28a745;margin-bottom:8px;'>🚀 Operations</div>"),
        action_components['container'],
        widgets.HTML("<div style='margin:8px 0 4px 0;font-size:13px;color:#666;'><strong>📋 Konfirmasi & Status:</strong></div>"),
        confirmation_area
    ], layout=widgets.Layout(
        width='100%', 
        margin='10px 0', 
        padding='12px',
        border='1px solid #e0e0e0', 
        border_radius='8px',
        height='auto',
        background_color='#f9f9f9'
    ))
    
    # === MAIN UI ASSEMBLY ===
    
    ui = widgets.VBox([
        header,
        status_panel,
        input_options,
        config_section,
        action_section,
        progress_container,
        log_components['log_accordion']
    ], layout=widgets.Layout(
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='15px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        box_shadow='0 2px 4px rgba(0,0,0,0.05)'
    ))
    
    # === COMPONENT EXTRACTION ===
    
    # Safe extraction dengan fallback
    def safe_extract(obj, attr, fallback=None):
        try:
            return getattr(obj, attr, fallback)
        except (AttributeError, TypeError):
            return fallback
    
    # Debug: Print action_components structure untuk debugging
    print(f"[DEBUG] action_components keys: {list(action_components.keys()) if action_components else 'None'}")
    
    # Build UI components dictionary
    ui_components.update({
        # CRITICAL COMPONENTS (required by CommonInitializer)  
        'ui': ui,
        'preprocess_button': (action_components.get('preprocess_btn') or 
                             action_components.get('primary_button') or
                             action_components.get('download_button')),  # Fallback options
        'check_button': (action_components.get('check_btn') or
                        action_components.get('secondary_button') or
                        action_components.get('info_button')),  # Fallback options
        'cleanup_button': action_components.get('cleanup_btn'),
        'save_button': save_reset_components.get('save_button'),
        'reset_button': save_reset_components.get('reset_button'),
        'log_output': log_components.get('log_output'),
        'status_panel': status_panel,
        
        # UI SECTIONS
        'header': header,
        'input_options': input_options,
        'config_section': config_section,
        'action_section': action_section,
        'confirmation_area': confirmation_area,
        'dialog_area': confirmation_area,  # Alias untuk compatibility
        
        # PROGRESS TRACKING
        'progress_tracker': progress_tracker,
        'progress_container': progress_container,
        
        # LOG COMPONENTS
        'log_accordion': log_components.get('log_accordion'),
        
        # ACTION COMPONENTS
        'action_buttons': action_components,
        'save_reset_buttons': save_reset_components,
        
        # INPUT FORM COMPONENTS
        'resolution_dropdown': safe_extract(input_options, 'resolution_dropdown'),
        'normalization_dropdown': safe_extract(input_options, 'normalization_dropdown'),
        'preserve_aspect_checkbox': safe_extract(input_options, 'preserve_aspect_checkbox'),
        'target_splits_select': safe_extract(input_options, 'target_splits_select'),
        'batch_size_input': safe_extract(input_options, 'batch_size_input'),
        'validation_checkbox': safe_extract(input_options, 'validation_checkbox'),
        'move_invalid_checkbox': safe_extract(input_options, 'move_invalid_checkbox'),
        'invalid_dir_input': safe_extract(input_options, 'invalid_dir_input'),
        'cleanup_target_dropdown': safe_extract(input_options, 'cleanup_target_dropdown'),
        'backup_checkbox': safe_extract(input_options, 'backup_checkbox'),
        
        # METADATA
        'module_name': 'preprocessing',
        'ui_initialized': True,
        'data_dir': config.get('data', {}).get('dir', 'data'),
        'api_integration': True,
        'dialog_support': True,
        'progress_tracking': True
    })
    
    # Log missing components for debugging
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    
    return ui_components