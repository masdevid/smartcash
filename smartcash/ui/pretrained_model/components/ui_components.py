"""
File: smartcash/ui/pretrained_model/components/ui_components.py
Deskripsi: Fixed UI components dengan single progress tracker untuk avoid weak reference error
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.header import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.utils.layout_utils import create_divider, get_layout
from smartcash.ui.utils.constants import ICONS, COLORS

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained model UI dengan single progress tracker untuk avoid weak reference error"""
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
    
    # Single progress tracker untuk avoid weak reference error
    progress_tracker = _create_single_progress_tracker()
    
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
        progress_tracker['container'], log_components['log_accordion']
    ], layout=get_layout('container'))
    
    return {
        'ui': ui, 'header': header, 'status_panel': status_panel,
        'config_form': config_form, 'download_sync_button': download_button,
        'progress_tracker': progress_tracker, 'log_components': log_components,
        'log_output': log_components['log_output'], 'status': log_components['log_output'],
        'models_dir_input': config_form['models_dir'], 'drive_models_dir_input': config_form['drive_models_dir'],
        'yolov5_url_input': config_form['yolov5_url'], 'efficientnet_url_input': config_form['efficientnet_url'],
        'module_name': 'pretrained_model', 'auto_check_enabled': True,
        # Progress tracker methods untuk compatibility
        'show_for_operation': progress_tracker['show_for_operation'],
        'update_progress': progress_tracker['update_progress'],
        'complete_operation': progress_tracker['complete_operation'],
        'error_operation': progress_tracker['error_operation'],
        'reset_all': progress_tracker['reset_all']
    }

def _create_single_progress_tracker() -> Dict[str, Any]:
    """Create single progress tracker dengan clean implementation untuk avoid weak reference error"""
    
    # Progress bar untuk overall operation
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Progress:',
                                     bar_style='', style={'bar_color': '#007bff'},
                                     layout=widgets.Layout(width='100%', margin='5px 0'))
    
    # Progress message
    progress_message = widgets.HTML(value="<div style='margin: 5px 0; color: #666;'>Siap memulai operasi...</div>",
                                   layout=widgets.Layout(width='100%'))
    
    # Progress container dengan visibility control
    progress_container = widgets.VBox([progress_bar, progress_message],
                                     layout=widgets.Layout(width='100%', margin='10px 0', padding='10px',
                                                          border='1px solid #ddd', border_radius='4px',
                                                          visibility='hidden'))
    
    # Operation state
    operation_state = {'current_operation': None, 'is_visible': False}
    
    def show_for_operation(operation_name: str) -> None:
        """Show progress tracker untuk operation tertentu"""
        operation_state['current_operation'] = operation_name
        operation_state['is_visible'] = True
        progress_container.layout.visibility = 'visible'
        progress_bar.value = 0
        progress_message.value = f"<div style='margin: 5px 0; color: #007bff;'>🚀 Memulai {operation_name}...</div>"
    
    def update_progress(category: str, value: int, message: str) -> None:
        """Update progress dengan message"""
        if operation_state['is_visible']:
            progress_bar.value = min(max(value, 0), 100)
            progress_message.value = f"<div style='margin: 5px 0; color: #333;'>{message}</div>"
    
    def complete_operation(final_message: str) -> None:
        """Complete operation dengan success message"""
        if operation_state['is_visible']:
            progress_bar.value = 100
            progress_bar.bar_style = 'success'
            progress_message.value = f"<div style='margin: 5px 0; color: #28a745;'>✅ {final_message}</div>"
            # Auto hide setelah delay
            _auto_hide_after_delay()
    
    def error_operation(error_message: str) -> None:
        """Handle error operation dengan error styling"""
        if operation_state['is_visible']:
            progress_bar.bar_style = 'danger'
            progress_message.value = f"<div style='margin: 5px 0; color: #dc3545;'>❌ {error_message}</div>"
            # Auto hide setelah delay
            _auto_hide_after_delay()
    
    def reset_all() -> None:
        """Reset progress tracker ke initial state"""
        operation_state['current_operation'] = None
        operation_state['is_visible'] = False
        progress_container.layout.visibility = 'hidden'
        progress_bar.value = 0
        progress_bar.bar_style = ''
        progress_message.value = "<div style='margin: 5px 0; color: #666;'>Siap memulai operasi...</div>"
    
    def _auto_hide_after_delay() -> None:
        """Auto hide progress tracker setelah operation selesai"""
        import threading
        threading.Timer(3.0, lambda: reset_all()).start()
    
    return {
        'container': progress_container,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'show_for_operation': show_for_operation,
        'update_progress': update_progress,
        'complete_operation': complete_operation,
        'error_operation': error_operation,
        'reset_all': reset_all,
        'operation_state': operation_state
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