# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: Complete UI Components untuk pretrained module dengan fixed API usage
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional
from smartcash.ui.components.progress_tracker.factory import create_dual_progress_tracker
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.utils.header_utils import create_header
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
    🏗️ Create complete pretrained UI components dengan fixed API usage
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary berisi semua UI components
    """
    try:
        # 1. 📊 Progress Tracker - Fixed API usage
        progress_tracker = create_dual_progress_tracker(
            operation="Pretrained Models Setup",
            auto_hide=True
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
        
        # 7. 💬 Dialog Area - Fixed API usage  
        confirmation_area = widgets.Output(
            layout=widgets.Layout(
                width='100%', 
                min_height='50px', 
                max_height='200px',
                margin='10px 0',
                padding='5px',
                border='1px solid #e0e0e0',
                border_radius='4px',
                background_color='#fafafa'
            )
        )
        
        # 8. 🎛️ Input Options
        input_options = _create_input_options(config)
        
        # 9. 📝 Module Header
        header = create_header(
            title="🤖 Pretrained Models Setup",
            description="Setup dan sinkronisasi pretrained models untuk YOLO detection",
            module_name="pretrained"
        )
        
        # 10. 📦 Main Container Assembly
        main_container = widgets.VBox([
            header,
            status_panel,
            progress_tracker['container'],  # Fixed: access container properly
            input_options.get('container', widgets.HTML("Input options tidak tersedia")),
            action_buttons,
            save_reset_buttons['container'],
            log_accordion,
            confirmation_area
        ], layout=widgets.Layout(
            padding='10px',
            width='100%'
        ))
        
        # 11. 🔧 Return UI Components Dictionary
        ui_components = {
            'ui': main_container,
            'header': header,
            'status_panel': status_panel,
            'progress_tracker': progress_tracker['tracker'],  # Fixed: access tracker properly
            'progress_container': progress_tracker['container'],
            'log_output': log_output,
            'log_accordion': log_accordion,
            'download_sync_button': download_sync_button,
            'action_buttons': action_buttons,
            'save_button': save_button,
            'reset_button': reset_button,
            'save_reset_buttons': save_reset_buttons,
            'confirmation_area': confirmation_area,  # Fixed: proper dialog area
            'input_options': input_options,
            
            # Individual input widgets
            'models_dir_input': input_options.get('models_dir_input'),
            'drive_models_dir_input': input_options.get('drive_models_dir_input'),
            'pretrained_type_dropdown': input_options.get('pretrained_type_dropdown'),
            'auto_download_checkbox': input_options.get('auto_download_checkbox'),
            'sync_drive_checkbox': input_options.get('sync_drive_checkbox'),
            
            # Metadata
            'module_name': 'pretrained_models',
            'config': config,
            'created_at': 'UI components created successfully'
        }
        
        logger.info("✅ Complete pretrained UI components created dengan fixed API")
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
            description='☁️ Sync with Drive',
            style={'description_width': '120px'}
        )
        
        # Container untuk input options
        container = widgets.VBox([
            widgets.HTML("<h4>📝 Configuration Options</h4>"),
            models_dir_input,
            drive_models_dir_input,
            widgets.HBox([pretrained_type_dropdown, auto_download_checkbox, sync_drive_checkbox])
        ], layout=widgets.Layout(
            padding='10px',
            border='1px solid #ddd',
            border_radius='5px',
            margin='10px 0'
        ))
        
        return {
            'container': container,
            'models_dir_input': models_dir_input,
            'drive_models_dir_input': drive_models_dir_input,
            'pretrained_type_dropdown': pretrained_type_dropdown,
            'auto_download_checkbox': auto_download_checkbox,
            'sync_drive_checkbox': sync_drive_checkbox
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating input options: {str(e)}")
        return {'container': widgets.HTML(f"Error: {str(e)}")}

def _create_fallback_ui(error_msg: str) -> Dict[str, Any]:
    """🚨 Create fallback UI jika main creation gagal"""
    fallback_container = widgets.VBox([
        widgets.HTML(f"<h3>❌ Error: Pretrained UI Components</h3>"),
        widgets.HTML(f"<p>{error_msg}</p>"),
        widgets.HTML("<p>Menggunakan fallback UI minimal.</p>")
    ], layout=widgets.Layout(
        padding='20px',
        border='2px solid red',
        border_radius='5px'
    ))
    
    return {
        'ui': fallback_container,
        'status_panel': widgets.HTML("Error state"),
        'log_output': widgets.Output(),
        'download_sync_button': widgets.Button(description="Error", disabled=True),
        'save_button': widgets.Button(description="Save", disabled=True),
        'reset_button': widgets.Button(description="Reset", disabled=True),
        'confirmation_area': widgets.Output(),
        'error': True,
        'error_message': error_msg
    }