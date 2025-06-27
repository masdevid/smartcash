"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Fixed UI components dengan proper progress tracker integration dan button mapping
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create preprocessing UI dengan fixed progress tracker dan proper button mapping"""
    config = config or {}
    
    # Initialize UI components dictionary
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
        "Dataset Preprocessing", 
        "Preprocessing dataset dengan YOLO normalization dan real-time progress",
        "🚀"
    )
    
    # Status panel
    status_panel = create_status_panel(
        "🚀 Siap memulai preprocessing", 
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
    
    # Action components - FIXED: Use proper button IDs
    action_components = create_action_buttons(
        primary_button={
            "label": "🚀 Mulai Preprocessing",
            "style": "success",
            "width": "180px"
        },
        secondary_buttons=[
            {
                "label": "🔍 Check Dataset",
                "style": "info",
                "width": "150px"
            },
            {
                "label": "🗑️ Cleanup",
                "style": "warning",
                "width": "120px"
            }
        ]
    )
    
    # Progress tracker - FIXED: Proper initialization
    progress_tracker = create_dual_progress_tracker(
        operation="Dataset Preprocessing",
        auto_hide=False
    )
    
    # CRITICAL: Show progress tracker initially untuk ensure proper attachment
    if hasattr(progress_tracker, 'show'):
        progress_tracker.show()
    
    # Log components
    log_components = create_log_accordion(
        module_name='preprocessing',
        height='200px'
    )
    
    # Confirmation area
    confirmation_area, _ = create_confirmation_area()
    ui_components['confirmation_area'] = confirmation_area
    
    # === LAYOUT SECTIONS ===
    
    # Config section
    config_section = widgets.VBox([
        widgets.Box([save_reset_components['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(margin='8px 0'))
    
    # Action section with flex layout
    action_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#28a745;margin-bottom:8px;'>🚀 Operations</div>"),
        action_components['container'],
        widgets.HTML("<div style='margin:8px 0 4px 0;font-size:13px;color:#666;'><strong>📋 Status:</strong></div>"),
        widgets.Box(
            [confirmation_area],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='space-between',
                align_items='center',
                width='100%',
                margin='0',
                padding='0'
            )
        )
    ], layout=widgets.Layout(
        display='flex',
        flex_direction='column',
        width='100%',
        margin='10px 0',
        padding='12px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#f9f9f9',
        overflow='hidden'
    ))
    
    # === MAIN UI ASSEMBLY dengan Progress Tracker ===
    
    ui = widgets.VBox([
        header,
        status_panel,
        input_options,
        config_section,
        action_section,
        # FIXED: Ensure progress tracker container exists dan visible
        progress_tracker.container if hasattr(progress_tracker, 'container') else widgets.VBox([]),
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
    
    # === FIXED BUTTON MAPPING ===
    
    # Extract buttons dengan nama yang konsisten
    preprocess_btn = action_components.get('primary') or action_components.get('mulai_preprocessing')
    check_btn = action_components.get('secondary_0') or action_components.get('check_dataset')
    cleanup_btn = action_components.get('secondary_1') or action_components.get('cleanup')
    
    # CRITICAL: Map dengan nama yang dicari handler
    ui_components.update({
        # REQUIRED COMPONENTS untuk CommonInitializer
        'ui': ui,
        'log_output': log_components.get('log_output'),
        'status_panel': status_panel,
        
        # BUTTONS dengan nama yang dicari handler
        'preprocess_btn': preprocess_btn,
        'check_btn': check_btn, 
        'cleanup_btn': cleanup_btn,
        'save_button': save_reset_components.get('save_button'),
        'reset_button': save_reset_components.get('reset_button'),
        
        # ALIASES untuk backward compatibility
        'preprocess_button': preprocess_btn,
        'check_button': check_btn,
        'cleanup_button': cleanup_btn,
        
        # UI SECTIONS
        'header': header,
        'input_options': input_options,
        'config_section': config_section,
        'action_section': action_section,
        'confirmation_area': confirmation_area,
        
        # PROGRESS & LOG - FIXED: Proper progress tracker attachment
        'progress_tracker': progress_tracker,
        'progress': progress_tracker,  # Alias untuk compatibility
        'log_accordion': log_components.get('log_accordion'),
        
        # ACTION COMPONENTS
        'action_buttons': {
            'preprocess_btn': preprocess_btn,
            'check_btn': check_btn,
            'cleanup_btn': cleanup_btn
        },
        'save_reset_buttons': save_reset_components,
        
        # INPUT FORM COMPONENTS
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None),
        'preserve_aspect_checkbox': getattr(input_options, 'preserve_aspect_checkbox', None),
        'target_splits_select': getattr(input_options, 'target_splits_select', None),
        'batch_size_input': getattr(input_options, 'batch_size_input', None),
        'validation_checkbox': getattr(input_options, 'validation_checkbox', None),
        'move_invalid_checkbox': getattr(input_options, 'move_invalid_checkbox', None),
        'invalid_dir_input': getattr(input_options, 'invalid_dir_input', None),
        'cleanup_target_dropdown': getattr(input_options, 'cleanup_target_dropdown', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        
        # METADATA
        'module_name': 'preprocessing',
        'ui_initialized': True,
        'api_integration': True
    })
    
    return ui_components