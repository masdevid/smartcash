# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models - Simplified dengan YOLOv5s only
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.common.logger import get_logger
from smartcash.ui.components import (
    create_header,
    create_status_panel,
    create_action_buttons,
    create_log_accordion,
    create_dual_progress_tracker
)

logger = get_logger(__name__)

def create_pretrained_main_ui(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """🎯 Create main UI untuk pretrained models dengan simplified approach
    
    Args:
        config: Konfigurasi untuk inisialisasi UI
        **kwargs: Parameter tambahan
            
    Returns:
        Dictionary berisi komponen UI yang dibuat
    """
    try:
        pretrained_config = config.get('pretrained_models', {})
        
        # Create input options
        input_options = create_pretrained_input_options(pretrained_config)
        
        # Header
        header = create_header(
            title="🤖 Pretrained Models Configuration",
            description="Setup YOLOv5s untuk currency detection",
            icon="🎯"
        )
        
        # Status panel
        status_panel = create_status_panel()
        
        # Create action buttons container
        action_buttons = create_action_buttons(
            primary_button={"label": "📥 Download & Sync", "style": "primary"},
            secondary_buttons=[
                {"label": "💾 Save Config", "style": "success"},
                {"label": "🔄 Reset", "style": "warning"}
            ]
        )
        
        # Get individual button references
        download_btn = action_buttons.get('download_sync', action_buttons.get('buttons', {}).get('download_sync'))
        save_btn = action_buttons.get('save', action_buttons.get('buttons', {}).get('save'))
        reset_btn = action_buttons.get('reset', action_buttons.get('buttons', {}).get('reset'))
        
        # Ensure we have all buttons
        if not all([download_btn, save_btn, reset_btn]):
            # Fallback to first three buttons if direct access fails
            all_buttons = action_buttons.get('buttons', [])
            if isinstance(all_buttons, dict):
                all_buttons = list(all_buttons.values())
            if len(all_buttons) >= 3:
                download_btn, save_btn, reset_btn = all_buttons[:3]
        
        # Progress tracker
        progress_tracker = create_dual_progress_tracker()
        
        # Create log accordion and get its components
        log_components = create_log_accordion(
            module_name="Pretrained Models",
            height="200px"
        )
        log_accordion = log_components['log_accordion']
        log_output = log_components['log_output']
        
        # Update log_components with the correct references
        log_components.update({
            'ui': log_accordion,
            'output': log_output
        })
        
        # Create main layout
        main_ui = widgets.VBox([
            header,
            input_options['ui'],
            action_buttons['container'],
            status_panel,
            progress_tracker['ui'] if isinstance(progress_tracker, dict) else progress_tracker,
            log_output
        ])
        
        # Store all components in dictionary with required 'ui' key
        ui_components = {
            'ui': main_ui,  # Main UI component
            'header': header,
            'status_panel': status_panel,
            'action_buttons': action_buttons,
            'progress_tracker': progress_tracker,
            'log_output': log_output,
            'log_accordion': log_accordion,
            
            # Input components
            **input_options,
            
            # Action button references
            'download_sync_button': download_btn,
            'save_button': save_btn,
            'reset_button': reset_btn,
            
            # Status components
            'status': status_panel.get('status', status_panel),
            'confirmation_area': status_panel.get('confirmation_area'),
            
            # Progress components
            'main_progress': progress_tracker['main_progress'],
            'sub_progress': progress_tracker['sub_progress'],
            
            # State flags
            'pretrained_initialized': True,
            'ui_initialized': True
        }
        
        logger.info("✅ Pretrained UI components berhasil dibuat")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error creating pretrained UI: {str(e)}")
        raise


def create_pretrained_input_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """🔧 Create input options untuk pretrained configuration
    
    Args:
        config: Konfigurasi pretrained models
        
    Returns:
        Dictionary berisi input components
    """
    try:
        # Models directory input
        models_dir_input = widgets.Text(
            value=config.get('models_dir', '/content/models'),
            description='Models Dir:',
            placeholder='/content/models',
            style={'description_width': '120px'},
            layout={'width': '400px'}
        )
        
        # Drive models directory input - Updated path
        drive_models_dir_input = widgets.Text(
            value=config.get('drive_models_dir', '/data/pretrained'),
            description='Drive Dir:',
            placeholder='/data/pretrained',
            style={'description_width': '120px'},
            layout={'width': '400px'}
        )
        
        # Pretrained type - Simplified to YOLOv5s only
        pretrained_type_info = widgets.HTML(
            value="<b>🤖 Model Type:</b> YOLOv5s (Optimal untuk currency detection)"
        )
        
        # Hidden dropdown for consistency (always yolov5s)
        pretrained_type_dropdown = widgets.Dropdown(
            options=['yolov5s'],
            value='yolov5s',
            description='Model Type:',
            style={'description_width': '120px'},
            layout={'width': '300px', 'display': 'none'}  # Hidden
        )
        
        # Auto download checkbox
        auto_download_checkbox = widgets.Checkbox(
            value=config.get('auto_download', False),
            description='Auto Download',
            style={'description_width': '120px'}
        )
        
        # Sync drive checkbox
        sync_drive_checkbox = widgets.Checkbox(
            value=config.get('sync_drive', True),
            description='Sync to Drive',
            style={'description_width': '120px'}
        )
        
        # Layout input options
        input_ui = widgets.VBox([
            widgets.HTML("<h4>📁 Directory Configuration</h4>"),
            models_dir_input,
            drive_models_dir_input,
            widgets.HTML("<br>"),
            pretrained_type_info,
            widgets.HTML("<h4>⚙️ Options</h4>"),
            widgets.HBox([auto_download_checkbox, sync_drive_checkbox])
        ])
        
        return {
            'ui': input_ui,
            'models_dir_input': models_dir_input,
            'drive_models_dir_input': drive_models_dir_input,
            'pretrained_type_dropdown': pretrained_type_dropdown,
            'auto_download_checkbox': auto_download_checkbox,
            'sync_drive_checkbox': sync_drive_checkbox
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating input options: {str(e)}")
        raise


# Remove fallback UI function - errors should be handled by CommonInitializer