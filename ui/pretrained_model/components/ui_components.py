"""
File: smartcash/ui/pretrained_model/components/ui_components.py
Deskripsi: Fixed UI components menggunakan existing progress_tracker.py untuk avoid weak reference error
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.header import create_header
from smartcash.ui.utils.layout_utils import create_divider, get_layout
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.common.logger import get_logger

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained model UI dengan existing progress tracker"""
    logger = get_logger('pretrained_model')
    logger.info("🔧 Memulai pembuatan UI components")
    
    config = config or {}
    
    # Pastikan modul progress_tracker tersedia
    try:
        from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
        logger.info("✅ Modul progress_tracker berhasil diimpor")
    except ImportError as e:
        logger.error(f"💥 Gagal mengimpor modul progress_tracker: {str(e)}")
        raise
    
    # Header
    header = create_header(f"{ICONS.get('model', '🤖')} Persiapan Model Pre-trained", 
                          "Download dan sinkronisasi model YOLOv5 dan EfficientNet-B4 untuk SmartCash")
    
    # Model configuration form
    config_form = _create_model_config_form(config)
    
    # Action button
    download_button = widgets.Button(description="Download & Sync Model", button_style='primary', 
                                   icon='download', layout=get_layout('responsive_button', width='200px'))
    
    # Save/Reset buttons
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    save_reset_components = create_save_reset_buttons(save_label="Simpan", reset_label="Reset")
    
    # Status panel
    from smartcash.ui.components.status_panel import create_status_panel
    status_panel = create_status_panel("Siap untuk download dan sinkronisasi model", "info")
    
    # Log output
    log_output = widgets.Output(layout=widgets.Layout(width='100%', height='200px', 
                                                     border='1px solid #ddd', overflow='auto',
                                                     padding='10px', margin='10px 0'))
    
    # Create progress tracker menggunakan dual progress
    try:
        progress_tracker = create_dual_progress_tracker(
            operation="Download Model",
            primary_label='Progress Keseluruhan', 
            secondary_label='Progress Saat Ini'
        )
    except Exception as e:
        logger = get_logger('pretrained_model')
        logger.error(f"💥 Gagal membuat progress tracker: {str(e)}")
        raise
    
    # Button group
    button_group = widgets.VBox([
        download_button
    ], layout=widgets.Layout(width='100%', margin='5px 0'))
    
    # Section headers
    action_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                           border-bottom: 2px solid {COLORS.get('primary', '#007bff')}; padding-bottom: 6px;'>
                                       {ICONS.get('play', '▶️')} Actions</h4>""")
       
    log_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                         border-bottom: 2px solid {COLORS.get('info', '#17a2b8')}; padding-bottom: 6px;'>
                                   {ICONS.get('log', '📋')} Log Aktivitas</h4>""")
    
    # Main UI assembly
    ui = widgets.VBox([
        header,
        status_panel,
        config_form['container'], 
        save_reset_components['container'],
        action_header, 
        button_group,
        progress_tracker.container,  
        log_header, 
        log_output
    ], layout=get_layout('container'))
    
    # Combine all components dengan progress tracker methods
    ui_components = {
        'ui': ui, 'header': header, 'config_form': config_form, 
        'download_sync_button': download_button,
        'save_button': save_reset_components['save_button'], 
        'reset_button': save_reset_components['reset_button'],
        'save_reset_components': save_reset_components,
        'status_panel': status_panel,
        'log_output': log_output,
        'models_dir_input': config_form['models_dir'], 
        'drive_models_dir_input': config_form['drive_models_dir'],
        'yolov5_url_input': config_form['yolov5_url'], 
        'efficientnet_url_input': config_form['efficientnet_url'],
        'module_name': 'pretrained_model', 
        'auto_check_enabled': True,
        # Progress tracker dengan API yang benar
        'progress_tracker': progress_tracker,
        'tracker': progress_tracker,
        'update_primary': progress_tracker.update_primary,
        'update_overall': progress_tracker.update_overall,
        'update_current': progress_tracker.update_current,
        'update_progress': progress_tracker.update,
        'complete_operation': progress_tracker.complete,
        'error_operation': progress_tracker.error,
        'reset_progress': progress_tracker.reset,
        'show_progress': progress_tracker.show
    }
    
    return ui_components

def _create_model_config_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model configuration form dengan explicit layout"""
    pretrained_config = config.get('pretrained_models', {})
    models_config = pretrained_config.get('models', {})
    
    # Directory inputs
    models_dir = widgets.Text(
        value=pretrained_config.get('models_dir', '/content/models'),
        description='Models Dir:', 
        placeholder='Path untuk menyimpan model lokal',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': '100px'}
    )
    
    drive_models_dir = widgets.Text(
        value=pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'),
        description='Drive Dir:', 
        placeholder='Path untuk sinkronisasi ke Drive',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': '100px'}
    )
    
    # Model URL inputs
    yolov5_config = models_config.get('yolov5', {})
    yolov5_url = widgets.Text(
        value=yolov5_config.get('url', 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt'),
        description='YOLOv5 URL:', 
        placeholder='URL download YOLOv5',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': '100px'}
    )
    
    efficientnet_config = models_config.get('efficientnet_b4', {})
    efficientnet_url = widgets.Text(
        value=efficientnet_config.get('url', 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin'),
        description='EfficientNet URL:', 
        placeholder='URL download EfficientNet-B4',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': '100px'}
    )
    
    # Form header
    form_header = widgets.HTML(f"""<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; 
                                          border-bottom: 2px solid {COLORS.get('info', '#17a2b8')}; padding-bottom: 6px;'>
                                     {ICONS.get('settings', '⚙️')} Konfigurasi Model</h4>""")
    
    # Form content
    form_content = widgets.VBox([
        models_dir, drive_models_dir, yolov5_url, efficientnet_url
    ], layout=widgets.Layout(width='100%'))
    
    # Container
    container = widgets.VBox([
        form_header, form_content
    ], layout=widgets.Layout(
        width='100%', margin='10px 0', padding='15px',
        border='1px solid #ddd', border_radius='4px'
    ))
    
    return {
        'container': container, 
        'models_dir': models_dir, 
        'drive_models_dir': drive_models_dir,
        'yolov5_url': yolov5_url, 
        'efficientnet_url': efficientnet_url, 
        'form_header': form_header
    }