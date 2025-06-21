# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: Complete UI Components untuk pretrained module dengan header integration
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional
from smartcash.ui.components.progress_tracker.factory import create_progress_tracker
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.dialog.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.utils.header_utils import create_module_header
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def safe_extract(obj: Optional[Dict], key: str, default=None):
    """🛡️ Safe extraction dengan null checks"""
    try:
        return obj.get(key, default) if obj else default
    except (AttributeError, TypeError):
        return default

def create_pretrained_ui_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    🏗️ Create complete pretrained UI components dengan header
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary berisi semua UI components
    """
    try:
        # 1. 📊 Progress Tracker
        progress_tracker = create_progress_tracker(
            tracker_id="pretrained_progress",
            show_overall=True,
            show_step=True,
            show_current=False
        )
        
        # 2. 📋 Status Panel
        status_panel = create_status_panel(
            initial_message="🔄 Ready untuk setup pretrained models",
            panel_type="info"
        )
        
        # 3. 📝 Log Output Areas
        log_output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='100px',
                max_height='300px',
                border='1px solid #ddd',
                padding='10px'
            )
        )
        
        # 4. 📂 Log Accordion
        log_accordion = create_log_accordion(
            log_output=log_output,
            title="📋 Pretrained Setup Logs",
            selected_index=None
        )
        
        # 5. 🎯 Action Buttons
        download_sync_button = widgets.Button(
            description="📥 Download & Sync Models",
            button_style='primary',
            icon='download',
            layout=widgets.Layout(width='200px', height='35px')
        )
        
        action_buttons = create_action_buttons([download_sync_button])
        
        # 6. 💾 Save & Reset Buttons
        save_reset_buttons = create_save_reset_buttons()
        save_button = save_reset_buttons['save_button']
        reset_button = save_reset_buttons['reset_button']
        
        # 7. 💬 Dialog Areas
        confirmation_area = widgets.Output(
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
        
        dialog_area = create_confirmation_dialog()
        
        # 8. 🎛️ Input Options
        input_options = _create_input_options(config)
        
        # 9. 📝 Module Header
        header = create_module_header(
            title="🤖 Pretrained Models Setup",
            description="Setup dan sinkronisasi pretrained models untuk YOLO detection",
            module_name="pretrained"
        )
        
        # 10. 📦 Main Container Assembly
        main_container = widgets.VBox([
            header,
            status_panel,
            progress_tracker.get('ui', widgets.HTML("Progress tracker tidak tersedia")),
            input_options.get('container', widgets.HTML("Input options tidak tersedia")),
            action_buttons,
            save_reset_buttons['container'],
            log_accordion,
            confirmation_area
        ], layout=widgets.Layout(
            padding='10px',
            width='100%'
        ))
        
        # 11. ✅ Complete Component Dictionary
        ui_components = {
            # Core UI Elements
            'ui': main_container,
            'main_container': main_container,
            'header': header,
            'status_panel': status_panel,
            
            # Progress Tracking
            'progress_tracker': progress_tracker,
            'progress_ui': progress_tracker.get('ui'),
            
            # Logging
            'log_output': log_output,
            'log_accordion': log_accordion,
            
            # Buttons
            'download_sync_button': download_sync_button,
            'save_button': save_button,
            'reset_button': reset_button,
            'action_buttons': action_buttons,
            
            # Dialog & Confirmation
            'confirmation_area': confirmation_area,
            'dialog_area': dialog_area,
            
            # Input Components
            'models_dir_input': safe_extract(input_options, 'models_dir_input'),
            'drive_models_dir_input': safe_extract(input_options, 'drive_models_dir_input'),
            'pretrained_type_dropdown': safe_extract(input_options, 'pretrained_type_dropdown'),
            'auto_download_checkbox': safe_extract(input_options, 'auto_download_checkbox'),
            'sync_drive_checkbox': safe_extract(input_options, 'sync_drive_checkbox'),
            'input_container': safe_extract(input_options, 'container'),
            
            # Metadata & Status
            'module_name': 'pretrained',
            'ui_initialized': True,
            'models_dir': config.get('pretrained_models', {}).get('models_dir', '/content/models'),
            'api_integration': True,
            'dialog_support': True,
            'progress_tracking': True,
            'error': None
        }
        
        logger.info("✅ Complete pretrained UI components created dengan header")
        return ui_components
        
    except Exception as e:
        error_msg = f"Gagal membuat UI components untuk pretrained: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return _create_fallback_ui(error_msg)

def _create_input_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """🎛️ Create input options widgets"""
    try:
        pretrained_config = config.get('pretrained_models', {})
        
        # Models directory input
        models_dir_input = widgets.Text(
            value=pretrained_config.get('models_dir', '/content/models'),
            description='📁 Models Dir:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Drive models directory input  
        drive_models_dir_input = widgets.Text(
            value=pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/models'),
            description='☁️ Drive Dir:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Pretrained type dropdown
        pretrained_type_dropdown = widgets.Dropdown(
            options=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
            value=pretrained_config.get('pretrained_type', 'yolov5s'),
            description='🔧 Type:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Auto download checkbox
        auto_download_checkbox = widgets.Checkbox(
            value=pretrained_config.get('auto_download', True),
            description='📥 Auto Download',
            style={'description_width': '120px'}
        )
        
        # Sync drive checkbox
        sync_drive_checkbox = widgets.Checkbox(
            value=pretrained_config.get('sync_drive', False),
            description='☁️ Sync to Drive',
            style={'description_width': '120px'}
        )
        
        # Header untuk config section
        config_header = widgets.HTML(
            value="<h4 style='margin:10px 0;color:#2c3e50;'>⚙️ Configuration</h4>"
        )
        
        # Layout container dengan proper spacing
        container = widgets.VBox([
            config_header,
            widgets.VBox([
                models_dir_input,
                drive_models_dir_input
            ], layout=widgets.Layout(margin='10px 0')),
            widgets.HBox([
                pretrained_type_dropdown, 
                auto_download_checkbox, 
                sync_drive_checkbox
            ], layout=widgets.Layout(margin='10px 0'))
        ], layout=widgets.Layout(
            border='1px solid #e1e8ed',
            border_radius='8px',
            padding='15px',
            margin='10px 0'
        ))
        
        return {
            'container': container,
            'models_dir_input': models_dir_input,
            'drive_models_dir_input': drive_models_dir_input,
            'pretrained_type_dropdown': pretrained_type_dropdown,
            'auto_download_checkbox': auto_download_checkbox,
            'sync_drive_checkbox': sync_drive_checkbox,
            'config_header': config_header
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Error creating input options: {str(e)}")
        fallback_html = widgets.HTML(f"""
            <div style='padding:15px;border:1px solid #f39c12;background:#fef9e7;border-radius:8px;'>
                ⚠️ Input options error: {str(e)}
            </div>
        """)
        return {'container': fallback_html}

def _create_fallback_ui(error_msg: str) -> Dict[str, Any]:
    """🚨 Create minimal fallback UI dengan header"""
    try:
        # Fallback header
        fallback_header = widgets.HTML(f"""
            <div style='padding:15px;background:#dc3545;color:white;border-radius:8px;margin-bottom:10px;'>
                <h3 style='margin:0;'>⚠️ Pretrained Models Setup (Error Mode)</h3>
                <p style='margin:5px 0 0 0;font-size:14px;'>UI initialization failed</p>
            </div>
        """)
        
        error_widget = widgets.HTML(f"""
            <div style='padding:20px;border:2px solid #dc3545;border-radius:8px;background:#f8d7da;color:#721c24;'>
                <h4>⚠️ UI Creation Error</h4>
                <p>{error_msg}</p>
                <small>💡 Coba restart cell atau check dependencies</small>
            </div>
        """)
        
        log_output = widgets.Output()
        confirmation_area = widgets.Output()
        
        # Mock progress tracker
        mock_progress_tracker = {
            'ui': widgets.HTML("<div style='color:#dc3545;'>❌ Progress tracker tidak tersedia</div>"),
            'show': lambda *args: None,
            'update_overall': lambda *args: None,
            'complete': lambda *args: None,
            'error': lambda *args: None
        }
        
        # Fallback main container
        main_container = widgets.VBox([
            fallback_header,
            error_widget,
            log_output
        ])
        
        return {
            # Core UI
            'ui': main_container,
            'main_container': main_container,
            'header': fallback_header,
            'status_panel': widgets.HTML(f"<div style='color:#dc3545;'>❌ {error_msg}</div>"),
            
            # Progress
            'progress_tracker': mock_progress_tracker,
            'progress_ui': mock_progress_tracker['ui'],
            
            # Logging
            'log_output': log_output,
            'log_accordion': widgets.Accordion(children=[log_output]),
            
            # Buttons - Disabled
            'download_sync_button': widgets.Button(description="❌ Error", disabled=True),
            'save_button': widgets.Button(description="❌ Error", disabled=True),
            'reset_button': widgets.Button(description="❌ Error", disabled=True),
            
            # Dialog
            'confirmation_area': confirmation_area,
            'dialog_area': confirmation_area,
            
            # Input placeholders
            'models_dir_input': None,
            'drive_models_dir_input': None,
            'pretrained_type_dropdown': None,
            'auto_download_checkbox': None,
            'sync_drive_checkbox': None,
            'input_container': None,
            
            # Metadata
            'module_name': 'pretrained',
            'ui_initialized': False,
            'error': error_msg
        }
        
    except Exception as fallback_error:
        # Ultimate fallback
        ultimate_fallback = widgets.HTML(f"""
            <div style='padding:20px;background:#dc3545;color:white;'>
                <h3>💥 Critical UI Error</h3>
                <p>Original: {error_msg}</p>
                <p>Fallback: {str(fallback_error)}</p>
            </div>
        """)
        
        return {
            'ui': ultimate_fallback,
            'main_container': ultimate_fallback,
            'header': ultimate_fallback,
            'module_name': 'pretrained',
            'ui_initialized': False,
            'error': f"Critical: {error_msg} | {str(fallback_error)}"
        }