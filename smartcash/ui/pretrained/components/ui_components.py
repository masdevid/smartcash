"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained module mengikuti pattern preprocessing UI
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained UI mengikuti pattern preprocessing UI"""
    config = config or {}
    
    try:
        # Import dengan error handling sesuai preprocessing pattern
        from smartcash.ui.utils.header_utils import create_header 
        from smartcash.ui.components.action_buttons import create_action_buttons
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
        from smartcash.ui.pretrained.components.input_options import create_pretrained_input_options
        
        # === CORE COMPONENTS ===
        
        # Header dengan pattern yang sama seperti preprocessing
        header = create_header(
            "🔽 Pretrained Models Setup", 
            "Download dan sync YOLOv5 + EfficientNet-B4 pretrained models",
            "🚀"
        )
        
        # Status panel dengan initial message yang sama
        status_panel = create_status_panel(
            "🚀 Siap memulai setup pretrained models", 
            "info"
        )
        
        # Input options (akan dibuat sesuai kebutuhan pretrained)
        input_options = create_pretrained_input_options(config)
        
        # Save/Reset buttons dengan pattern yang sama
        save_reset_components = create_save_reset_buttons(
            save_label="💾 Simpan",
            reset_label="🔄 Reset", 
            with_sync_info=True,
            sync_message="Konfigurasi akan disinkronkan dengan Drive"
        )
        
        # Action buttons untuk download/sync
        action_components = create_action_buttons(
            primary_label="🔽 Check, Download & Sync Models",
            primary_icon="download",
            secondary_buttons=[],
            cleanup_enabled=False,
            button_width='280px'
        )
        
        # Progress tracker
        progress_tracker = create_dual_progress_tracker(
            operation="Pretrained Models Setup",
            auto_hide=True
        )
        
        # Log accordion
        log_components = create_log_accordion(
            title="📝 Download & Sync Logs",
            auto_scroll=True,
            max_height='200px'
        )
        
        # === LAYOUT ASSEMBLY ===
        
        # Configuration section
        config_section = widgets.VBox([
            input_options.get('main_container', widgets.HTML("Input options loading...")),
            save_reset_components.get('container', widgets.HTML("Save/Reset loading..."))
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
        
        # Action section 
        action_section = widgets.VBox([
            action_components.get('main_container', widgets.HTML("Action buttons loading...")),
            progress_tracker.get('container', widgets.HTML("Progress loading..."))
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
        
        # Confirmation area (sama seperti preprocessing)
        confirmation_area = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                margin='10px 0',
                border='1px solid #ddd',
                padding='10px'
            )
        )
        
        # Progress container
        progress_container = widgets.VBox([
            progress_tracker.get('container', widgets.HTML(""))
        ], layout=widgets.Layout(width='100%'))
        
        # Main UI container
        main_ui = widgets.VBox([
            header,
            status_panel,
            config_section,
            action_section,
            log_components.get('container', widgets.HTML("Log loading...")),
            confirmation_area
        ], layout=widgets.Layout(width='100%', padding='15px'))
        
        # === RETURN COMPONENTS ===
        
        # Helper function sama seperti preprocessing
        def safe_extract(source_dict: Dict[str, Any], key: str, default=None):
            """Safely extract dari nested dict"""
            return source_dict.get(key, default) if source_dict else default
        
        ui_components = {
            # MAIN UI
            'ui': main_ui,
            
            # CORE BUTTONS
            'download_sync_button': action_components.get('primary_button'), # Renamed untuk pretrained
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
            'dialog_area': confirmation_area,  # Alias
            
            # PROGRESS TRACKING
            'progress_tracker': progress_tracker,
            'progress_container': progress_container,
            
            # LOG COMPONENTS
            'log_accordion': log_components.get('log_accordion'),
            
            # INPUT FORM COMPONENTS (pretrained specific)
            'yolov5_url_input': safe_extract(input_options, 'yolov5_url_input'),
            'efficientnet_url_input': safe_extract(input_options, 'efficientnet_url_input'),
            'models_dir_input': safe_extract(input_options, 'models_dir_input'),
            'drive_models_dir_input': safe_extract(input_options, 'drive_models_dir_input'),
            
            # METADATA
            'module_name': 'pretrained',
            'ui_initialized': True,
            'models_dir': config.get('pretrained_models', {}).get('models_dir', '/content/models'),
            'api_integration': True,
            'dialog_support': True,
            'progress_tracking': True
        }
        
        return ui_components
        
    except Exception as e:
        # Fallback UI sama seperti preprocessing
        error_msg = f"Error creating pretrained UI: {str(e)}"
        print(f"⚠️ {error_msg}")
        
        return _create_fallback_ui(error_msg)

def _create_fallback_ui(error_msg: str) -> Dict[str, Any]:
    """Create minimal fallback UI sama seperti preprocessing"""
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
        'download_sync_button': widgets.Button(description="Error", disabled=True),
        'save_button': widgets.Button(description="Error", disabled=True),
        'reset_button': widgets.Button(description="Error", disabled=True),
        'module_name': 'pretrained',
        'ui_initialized': False,
        'error': error_msg
    }