# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models - Fixed version dengan complete implementation
"""

# Standard library imports
import sys
import traceback
from typing import Dict, Any, Optional, List

# Third-party imports
import ipywidgets as widgets

# SmartCash imports
from smartcash.common.logger import get_logger
from smartcash.ui.components import (
    create_header,
    create_status_panel,
    create_action_buttons,
    create_log_accordion,
    create_dual_progress_tracker
)

logger = get_logger(__name__)

def create_pretrained_ui_components(env=None, config: Optional[Dict] = None, **kwargs) -> Dict:
    """🎯 Create pretrained UI menggunakan shared reusable components
    
    Args:
        env: Environment configuration (optional)
        config: Configuration dictionary (optional)
        **kwargs: Additional keyword arguments
            
    Returns:
        Dictionary berisi komponen UI atau fallback UI jika terjadi error
    """
    module_name = kwargs.get('module_name', 'pretrained_ui')
    logger = get_logger(module_name)
    
    # 🔧 Safe config handling
    if config is None:
        config = {}
    if not isinstance(config, dict):
        config = {}
    
    # Ensure pretrained_models section exists
    if 'pretrained_models' not in config:
        config['pretrained_models'] = {}
    
    pretrained_config = config.get('pretrained_models', {})
    
    # 📝 Create input options
    input_options = _create_pretrained_input_options(pretrained_config)
            
    # 🎨 Create UI components
    ui_components = {}
            
    # Header
    ui_components['header'] = create_header(
        title="🤖 Pretrained Models Configuration",
        subtitle="Setup YOLOv5 dan EfficientNet-B4 untuk deteksi mata uang"
    )
            
    # Status panel
    ui_components['status'] = create_status_panel()
            
    # Progress tracker
    ui_components['progress_tracker'] = create_dual_progress_tracker(
        primary_label="Model Download",
        secondary_label="Drive Sync"
    )
            
    # Action buttons with new API - using secondary_buttons for additional actions
    action_components = create_action_buttons(
        primary_label="Download & Sync Models",
        primary_icon="📥",
        secondary_buttons=[
            ("Simpan Config", "💾", "success"),
            ("Reset", "🔄", "warning")
        ],
        button_width='200px',
        primary_style='primary'
    )
    
    # Get buttons using new API
    download_button = action_components.get('primary_button')
    secondary_buttons = action_components.get('secondary_buttons', [])
    save_button = secondary_buttons[0] if len(secondary_buttons) > 0 else None
    reset_button = secondary_buttons[1] if len(secondary_buttons) > 1 else None
    
    # Fallback button creation if any button is missing
    if download_button is None:
        print("[WARNING] Download button not found, creating fallback")
        download_button = widgets.Button(description='📥 Download & Sync Models',
                                      button_style='primary')
        download_button.layout = widgets.Layout(width='200px')
    
    if save_button is None:
        print("[WARNING] Save button not found, creating fallback")
        save_button = widgets.Button(description='💾 Simpan Config',
                                  button_style='success')
        save_button.layout = widgets.Layout(width='150px')
    
    if reset_button is None:
        print("[WARNING] Reset button not found, creating fallback")
        reset_button = widgets.Button(description='🔄 Reset',
                                   button_style='warning')
        reset_button.layout = widgets.Layout(width='120px')
    
    # Update UI components with buttons for backward compatibility
    ui_components['download_sync_button'] = download_button
    ui_components['save_button'] = save_button
    ui_components['reset_button'] = reset_button
    
    # Store the container for layout
    ui_components['action_buttons_container'] = action_components.get('container', 
                                                                   widgets.HBox([download_button, save_button, reset_button]))
            
    # Log accordion
    ui_components['log_accordion'] = create_log_accordion()
    ui_components['log_output'] = ui_components['log_accordion']['log_output']
            
    # Dialog area (for confirmations)
    ui_components['confirmation_area'] = widgets.VBox(
        layout=widgets.Layout(display='none')
    )
            
    # 📋 Create main layout
            
    # Input form section
    input_form = widgets.VBox([
        widgets.HTML("<h4>📝 Konfigurasi Model</h4>"),
        input_options['models_dir_text'],
        input_options['drive_models_dir_text'],
        input_options['pretrained_type_dropdown'],
        input_options['auto_download_checkbox'],
        input_options['sync_drive_checkbox']
    ])
            
    # Action section
    action_section = widgets.HBox([
        ui_components['download_sync_button'],
        ui_components['save_button'],
        ui_components['reset_button']
    ])
            
    # Main container
    ui_components['main_container'] = widgets.VBox([
        ui_components['header'],
        ui_components['status'],
        input_form,
        action_section,
        ui_components['progress_tracker'],
        ui_components['confirmation_area'],
        ui_components['log_accordion'].get('accordion', ui_components['log_output'])
    ])
            
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    return ui_components

def _create_pretrained_input_options(pretrained_config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create input form widgets for pretrained models
    
    Args:
        pretrained_config: Configuration dictionary for pretrained models
        
    Returns:
        Dictionary containing widget components
    """
    # Ensure config is a dictionary
    config = pretrained_config if isinstance(pretrained_config, dict) else {}
    
    # Define model types
    model_types = [
        ('YOLOv5s (Ringan)', 'yolov5s'),
        ('YOLOv5m (Medium)', 'yolov5m'),
        ('YOLOv5l (Besar)', 'yolov5l'),
        ('YOLOv5x (Extra Besar)', 'yolov5x')
    ]
    
    # Get values from config with defaults
    model_type = config.get('pretrained_type', 'yolov5s')
    if model_type not in [t[1] for t in model_types]:
        model_type = 'yolov5s'
    
    # Create widgets
    widgets_dict = {
        'models_dir_text': widgets.Text(
            value=str(config.get('models_dir', '/content/models')),
            description='Models Dir:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='500px')
        ),
        'drive_models_dir_text': widgets.Text(
            value=str(config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')),
            description='Drive Models:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='500px')
        ),
        'pretrained_type_dropdown': widgets.Dropdown(
            options=model_types,
            value=model_type,
            description='Model Type:',
            style={'description_width': '120px'}
        ),
        'auto_download_checkbox': widgets.Checkbox(
            value=bool(config.get('auto_download', False)),
            description='Auto Download',
            style={'description_width': '120px'}
        ),
        'sync_drive_checkbox': widgets.Checkbox(
            value=bool(config.get('sync_drive', True)),
            description='Sync Drive',
            style={'description_width': '120px'}
        )
    }
    
    return widgets_dict