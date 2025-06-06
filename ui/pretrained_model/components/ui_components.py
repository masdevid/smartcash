"""
File: smartcash/ui/pretrained_model/components/ui_components.py
Deskripsi: UI components pretrained model dengan form konfigurasi dan progress tracker
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.header import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.utils.layout_utils import create_divider, get_layout
from smartcash.ui.utils.constants import ICONS, COLORS

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained model UI dengan form konfigurasi"""
    config = config or {}
    
    # Header
    header = create_header(f"{ICONS.get('model', '🤖')} Persiapan Model Pre-trained", 
                          "Download dan sinkronisasi model YOLOv5 dan EfficientNet-B4 untuk SmartCash")
    
    # Status panel
    status_panel = create_status_panel("Mempersiapkan konfigurasi model...", "info")
    
    # Model configuration form
    config_form = _create_model_config_form(config)
    
    # Action button
    download_button = widgets.Button(description="Download & Sync Model", button_style='primary', 
                                   icon='download', layout=get_layout('responsive_button', width='200px'))
    
    # Progress tracker
    progress_tracker = create_dual_progress_tracker(height='180px')
    
    # Log accordion
    log_components = create_log_accordion(module_name='pretrained_model', height='200px')
    
    # Action header
    action_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                           border-bottom: 2px solid {COLORS.get('primary', '#007bff')}; padding-bottom: 6px;'>
                                       {ICONS.get('play', '▶️')} Actions</h4>""")
    
    # Main UI assembly
    ui = widgets.VBox([
        header, status_panel, config_form['container'],
        create_divider(), action_header, download_button,
        progress_tracker.container, log_components['log_accordion']
    ], layout=get_layout('container'))
    
    return {
        'ui': ui, 'header': header, 'status_panel': status_panel,
        'config_form': config_form, 'download_sync_button': download_button,
        'progress_tracker': progress_tracker, 'log_components': log_components,
        'log_output': log_components['log_output'], 'status': log_components['log_output'],
        'models_dir_input': config_form['models_dir'], 'drive_models_dir_input': config_form['drive_models_dir'],
        'yolov5_url_input': config_form['yolov5_url'], 'efficientnet_url_input': config_form['efficientnet_url'],
        'module_name': 'pretrained_model', 'auto_check_enabled': True
    }

def _create_model_config_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model configuration form dengan one-liner style"""
    pretrained_config = config.get('pretrained_models', {})
    models_config = pretrained_config.get('models', {})
    
    # Directory inputs
    models_dir = widgets.Text(value=pretrained_config.get('models_dir', '/content/models'),
                             description='Models Dir:', placeholder='Path untuk menyimpan model lokal',
                             layout=get_layout('text_input'), style={'description_width': '100px'})
    
    drive_models_dir = widgets.Text(value=pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'),
                                   description='Drive Dir:', placeholder='Path untuk sinkronisasi ke Drive',
                                   layout=get_layout('text_input'), style={'description_width': '100px'})
    
    # Model URL inputs
    yolov5_config = models_config.get('yolov5', {})
    yolov5_url = widgets.Text(value=yolov5_config.get('url', 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt'),
                             description='YOLOv5 URL:', placeholder='URL download YOLOv5',
                             layout=get_layout('text_input'), style={'description_width': '100px'})
    
    efficientnet_config = models_config.get('efficientnet_b4', {})
    efficientnet_url = widgets.Text(value=efficientnet_config.get('url', 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin'),
                                   description='EfficientNet URL:', placeholder='URL download EfficientNet-B4',
                                   layout=get_layout('text_input'), style={'description_width': '100px'})
    
    # Form header
    form_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                          border-bottom: 2px solid {COLORS.get('info', '#17a2b8')}; padding-bottom: 6px;'>
                                     {ICONS.get('settings', '⚙️')} Konfigurasi Model</h4>""")
    
    # Two-column layout untuk form
    left_column = widgets.VBox([models_dir, yolov5_url], layout=get_layout('two_column_left'))
    right_column = widgets.VBox([drive_models_dir, efficientnet_url], layout=get_layout('two_column_right'))
    form_content = widgets.HBox([left_column, right_column], layout=get_layout('hbox'))
    
    container = widgets.VBox([form_header, form_content], 
                           layout=get_layout('card', margin='10px 0', padding='15px'))
    
    return {'container': container, 'models_dir': models_dir, 'drive_models_dir': drive_models_dir,
            'yolov5_url': yolov5_url, 'efficientnet_url': efficientnet_url, 'form_header': form_header}