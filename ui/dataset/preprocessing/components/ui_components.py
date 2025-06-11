"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Enhanced UI components dengan dialog integration dan progress tracking
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create preprocessing UI dengan API integration dan dialog support"""
    config = config or {}
    
    try:
        # Import dengan error handling
        from smartcash.ui.utils.header_utils import create_header 
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
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
            save_label="💾 Simpan",
            reset_label="🔄 Reset", 
            with_sync_info=True,
            sync_message="Konfigurasi akan disinkronkan dengan API"
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
        progress_components = create_dual_progress_tracker(
            operation="Dataset Preprocessing",
            auto_hide=False
        )
        
        # Log accordion
        log_components = create_log_accordion(
            module_name='preprocessing',
            height='200px'
        )
        
        # === DIALOG AREA ===
        
        # Confirmation/Dialog area untuk reusable dialogs
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', 
            min_height='50px', 
            max_height='250px',
            margin='10px 0',
            padding='5px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            background_color='#fafafa',
            overflow='auto'
        ))
        
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
            background_color='#f9f9f9'
        ))
        
        # === MAIN UI ASSEMBLY ===
        
        ui = widgets.VBox([
            header,
            status_panel,
            input_options,
            config_section,
            action_section,
            progress_components['container'],
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
        
        # Build UI components dictionary
        ui_components = {
            # CRITICAL COMPONENTS (required by CommonInitializer)
            'ui': ui,
            'preprocess_button': action_components.get('download_button'),  # primary button
            'check_button': action_components.get('check_button'),
            'cleanup_button': action_components.get('cleanup_button'),
            'save_button': save_reset_components.get('save_button'),
            'reset_button': save_reset_components.get('reset_button'),
            'log_output': log_components.get('log_output'),
            'status_panel': status_panel,
            
            # UI SECTIONS
            'header': header,
            'input_options': input_options,
            'config_section': config_section,
            'action_section': action_section,
            'confirmation_area': confirmation_area,  # Dialog area
            'dialog_area': confirmation_area,        # Alias untuk compatibility
            
            # PROGRESS TRACKING
            'progress_tracker': progress_components.get('tracker'),
            'progress_container': progress_components.get('container'),
            
            # LOG COMPONENTS
            'log_accordion': log_components.get('log_accordion'),
            
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
        }
        
        return ui_components
        
    except Exception as e:
        # Fallback UI jika terjadi error
        error_msg = f"Error creating preprocessing UI: {str(e)}"
        print(f"⚠️ {error_msg}")
        
        return _create_fallback_ui(error_msg)

def _create_fallback_ui(error_msg: str) -> Dict[str, Any]:
    """Create minimal fallback UI"""
    error_widget = widgets.HTML(f"""
        <div style='padding:20px;border:2px solid #dc3545;border-radius:8px;background:#f8d7da;color:#721c24;'>
            <h4>⚠️ UI Creation Error</h4>
            <p>{error_msg}</p>
            <small>💡 Try restarting the cell atau check dependencies</small>
        </div>
    """)
    
    log_output = widgets.Output()
    confirmation_area = widgets.Output()
    
    return {
        'ui': widgets.VBox([error_widget]),
        'status_panel': widgets.HTML(f"<div style='color:#dc3545;'>❌ {error_msg}</div>"),
        'log_output': log_output,
        'log_accordion': widgets.Accordion(children=[log_output]),
        'confirmation_area': confirmation_area,
        'dialog_area': confirmation_area,
        'progress_tracker': None,
        'preprocess_button': widgets.Button(description="Error", disabled=True),
        'check_button': widgets.Button(description="Error", disabled=True), 
        'cleanup_button': widgets.Button(description="Error", disabled=True),
        'save_button': widgets.Button(description="Error", disabled=True),
        'reset_button': widgets.Button(description="Error", disabled=True),
        'module_name': 'preprocessing',
        'ui_initialized': False,
        'error': error_msg
    }