"""
File: smartcash/ui/pretrained_model/components/ui_components.py
Deskripsi: Clean UI components tanpa dependencies yang berpotensi cause weak reference error
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.header import create_header
from smartcash.ui.utils.layout_utils import create_divider, get_layout
from smartcash.ui.utils.constants import ICONS, COLORS

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained model UI dengan clean implementation tanpa complex dependencies"""
    config = config or {}
    
    # Header
    header = create_header(f"{ICONS.get('model', '🤖')} Persiapan Model Pre-trained", 
                          "Download dan sinkronisasi model YOLOv5 dan EfficientNet-B4 untuk SmartCash")
    
   
    # Model configuration form
    config_form = _create_model_config_form(config)
    
    # Action button
    download_button = widgets.Button(description="Download & Sync Model", button_style='primary', 
                                   icon='download', layout=get_layout('responsive_button', width='200px'))
    
    # Save/Reset buttons menggunakan komponen yang sudah ada
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    save_reset_components = create_save_reset_buttons(save_label="Simpan", reset_label="Reset")
    
    # Button group dengan download dan save/reset
    button_group = widgets.VBox([
        download_button,
        
    ], layout=widgets.Layout(width='100%', margin='5px 0'))
    
    # Simple log output
    log_output = widgets.Output(layout=widgets.Layout(width='100%', height='200px', 
                                                     border='1px solid #ddd', overflow='auto',
                                                     padding='10px', margin='10px 0'))
    
    # Progress display
    progress_display = _create_simple_progress_display()
    
    # Action header
    action_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                           border-bottom: 2px solid {COLORS.get('primary', '#007bff')}; padding-bottom: 6px;'>
                                       {ICONS.get('play', '▶️')} Actions</h4>""")
    
    # Log header
    log_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                         border-bottom: 2px solid {COLORS.get('info', '#17a2b8')}; padding-bottom: 6px;'>
                                   {ICONS.get('log', '📋')} Log Aktivitas</h4>""")
    
    # Main UI assembly
    ui = widgets.VBox([
        header, config_form['container'], save_reset_components['container'], action_header, button_group,
        progress_display['container'], log_header, log_output
    ], layout=get_layout('container'))
    
    return {
        'ui': ui, 'header': header, 
        'config_form': config_form, 'download_sync_button': download_button,
        'save_button': save_reset_components['save_button'], 
        'reset_button': save_reset_components['reset_button'],
        'save_reset_components': save_reset_components,
        'log_output': log_output, 'progress_display': progress_display,
        'models_dir_input': config_form['models_dir'], 'drive_models_dir_input': config_form['drive_models_dir'],
        'yolov5_url_input': config_form['yolov5_url'], 'efficientnet_url_input': config_form['efficientnet_url'],
        'module_name': 'pretrained_model', 'auto_check_enabled': True,
        # Progress methods untuk compatibility
        'show_for_operation': progress_display['show_for_operation'],
        'update_progress': progress_display['update_progress'],
        'complete_operation': progress_display['complete_operation'],
        'error_operation': progress_display['error_operation'],
        'reset_all': progress_display['reset_all']
    }

def _create_simple_progress_display() -> Dict[str, Any]:
    """Create simple progress display tanpa complex widgets yang berpotensi error"""
    
    # Progress message display
    progress_html = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Progress container
    progress_container = widgets.VBox([progress_html], 
                                     layout=widgets.Layout(width='100%', visibility='hidden',
                                                          padding='10px', margin='5px 0',
                                                          border='1px solid #e9ecef', border_radius='4px'))
    
    # State tracking
    state = {'visible': False, 'current_operation': None}
    
    def show_for_operation(operation_name: str) -> None:
        """Show progress untuk operation"""
        state['current_operation'] = operation_name
        state['visible'] = True
        progress_container.layout.visibility = 'visible'
        progress_html.value = f"""<div style='color: #007bff; font-weight: bold;'>
                                   🚀 Memulai {operation_name}...</div>"""
    
    def update_progress(category: str, value: int, message: str) -> None:
        """Update progress message"""
        if state['visible']:
            # Simple progress bar menggunakan HTML
            bar_width = min(max(value, 0), 100)
            progress_html.value = f"""
            <div style='margin-bottom: 10px;'>
                <div style='background: #e9ecef; border-radius: 4px; overflow: hidden; height: 20px;'>
                    <div style='background: #007bff; height: 100%; width: {bar_width}%; transition: width 0.3s;'></div>
                </div>
                <div style='margin-top: 5px; color: #333; font-size: 14px;'>{message}</div>
                <div style='text-align: right; color: #666; font-size: 12px;'>{bar_width}%</div>
            </div>
            """
    
    def complete_operation(final_message: str) -> None:
        """Complete operation dengan success message"""
        if state['visible']:
            progress_html.value = f"""
            <div style='margin-bottom: 10px;'>
                <div style='background: #e9ecef; border-radius: 4px; overflow: hidden; height: 20px;'>
                    <div style='background: #28a745; height: 100%; width: 100%;'></div>
                </div>
                <div style='margin-top: 5px; color: #28a745; font-weight: bold;'>✅ {final_message}</div>
            </div>
            """
            # Auto hide setelah 3 detik
            import threading
            threading.Timer(3.0, reset_all).start()
    
    def error_operation(error_message: str) -> None:
        """Handle error dengan error styling"""
        if state['visible']:
            progress_html.value = f"""
            <div style='margin-bottom: 10px;'>
                <div style='background: #e9ecef; border-radius: 4px; overflow: hidden; height: 20px;'>
                    <div style='background: #dc3545; height: 100%; width: 100%;'></div>
                </div>
                <div style='margin-top: 5px; color: #dc3545; font-weight: bold;'>❌ {error_message}</div>
            </div>
            """
    
    def reset_all() -> None:
        """Reset progress display"""
        state['visible'] = False
        state['current_operation'] = None
        progress_container.layout.visibility = 'hidden'
        progress_html.value = ""
    
    return {
        'container': progress_container,
        'progress_html': progress_html,
        'show_for_operation': show_for_operation,
        'update_progress': update_progress,
        'complete_operation': complete_operation,
        'error_operation': error_operation,
        'reset_all': reset_all,
        'state': state
    }

def _create_model_config_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model configuration form dengan safe widget creation"""
    pretrained_config = config.get('pretrained_models', {})
    models_config = pretrained_config.get('models', {})
    
    # Directory inputs dengan explicit layout
    models_dir = widgets.Text(value=pretrained_config.get('models_dir', '/content/models'),
                             description='Models Dir:', placeholder='Path untuk menyimpan model lokal',
                             layout=widgets.Layout(width='100%', margin='5px 0'),
                             style={'description_width': '100px'})
    
    drive_models_dir = widgets.Text(value=pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'),
                                   description='Drive Dir:', placeholder='Path untuk sinkronisasi ke Drive',
                                   layout=widgets.Layout(width='100%', margin='5px 0'),
                                   style={'description_width': '100px'})
    
    # Model URL inputs dengan explicit layout
    yolov5_config = models_config.get('yolov5', {})
    yolov5_url = widgets.Text(value=yolov5_config.get('url', 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt'),
                             description='YOLOv5 URL:', placeholder='URL download YOLOv5',
                             layout=widgets.Layout(width='100%', margin='5px 0'),
                             style={'description_width': '100px'})
    
    efficientnet_config = models_config.get('efficientnet_b4', {})
    efficientnet_url = widgets.Text(value=efficientnet_config.get('url', 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin'),
                                   description='EfficientNet URL:', placeholder='URL download EfficientNet-B4',
                                   layout=widgets.Layout(width='100%', margin='5px 0'),
                                   style={'description_width': '100px'})
    
    # Form header
    form_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                          border-bottom: 2px solid {COLORS.get('info', '#17a2b8')}; padding-bottom: 6px;'>
                                     {ICONS.get('settings', '⚙️')} Konfigurasi Model</h4>""")
    
    # Simple vertical layout untuk avoid complex layout issues
    form_content = widgets.VBox([models_dir, drive_models_dir, yolov5_url, efficientnet_url],
                               layout=widgets.Layout(width='100%'))
    
    container = widgets.VBox([form_header, form_content], 
                           layout=widgets.Layout(width='100%', margin='10px 0', padding='15px',
                                                border='1px solid #ddd', border_radius='4px'))
    
    return {'container': container, 'models_dir': models_dir, 'drive_models_dir': drive_models_dir,
            'yolov5_url': yolov5_url, 'efficientnet_url': efficientnet_url, 'form_header': form_header}