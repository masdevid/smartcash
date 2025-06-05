"""
File: smartcash/ui/pretrained_model/components/ui_components.py
Deskripsi: UI components pretrained model dengan proper imports
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.components.header import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.pretrained_model.constants.model_constants import MODEL_CONFIGS, DEFAULT_MODELS_DIR, DEFAULT_DRIVE_MODELS_DIR

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained model UI components"""
    get_icon = lambda key, fallback="⚙️": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('model')} Persiapan Model Pre-trained", 
        "Download dan sinkronisasi model YOLOv5 dan EfficientNet-B4 untuk SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel("Mempersiapkan auto-check model...", "info")
    
    # Model info display
    models_dir = config.get('models_dir', DEFAULT_MODELS_DIR) if config else DEFAULT_MODELS_DIR
    drive_models_dir = config.get('drive_models_dir', DEFAULT_DRIVE_MODELS_DIR) if config else DEFAULT_DRIVE_MODELS_DIR
    
    model_list_html = ''.join([
        f"<li><b>{cfg['name']}</b> ({cfg['min_size_mb']} MB) - {cfg['description']}</li>"
        for cfg in MODEL_CONFIGS.values()
    ])
    
    info_html = widgets.HTML(f"""
        <div style='padding:10px; background-color:{COLORS['alert_info_bg']}; 
                   color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {COLORS['alert_info_text']}'>
            <p style='margin:5px 0'>{ICONS['info']} <b>Model yang akan diunduh:</b></p>
            <ul style='margin:5px 0'>{model_list_html}</ul>
            <p style='margin:8px 0'><b>Lokasi penyimpanan:</b></p>
            <ul style='margin:5px 0'>
                <li>Lokal: <code>{models_dir}</code></li>
                <li>Google Drive: <code>{drive_models_dir}</code></li>
            </ul>
            <p style='margin:5px 0; font-size:0.9em'>Auto-check akan berjalan setelah UI dimuat. Model existing akan di-skip.</p>
        </div>
    """)
    
    # Action buttons
    action_buttons = create_action_buttons(
        primary_label="Download & Sync Model",
        primary_icon="download",
        secondary_buttons=[("Reset UI", "refresh", "info")],
        cleanup_enabled=False,
        button_width='150px'
    )
    
    # Log accordion
    log_components = create_log_accordion(module_name='pretrained_model', height='220px')
    
    # Progress tracking dengan dual level untuk overall dan step progress
    progress_tracker_obj = create_dual_progress_tracker()
    
    # Membuat dictionary untuk backward compatibility
    progress_tracker = {
        'container': progress_tracker_obj.container,
        'progress_container': progress_tracker_obj.container,
        'status_widget': progress_tracker_obj.status_widget,
        'step_info_widget': progress_tracker_obj.step_info_widget,
        'tqdm_container': progress_tracker_obj.tqdm_container,
        'tracker': progress_tracker_obj,
        'show_container': progress_tracker_obj.show,
        'hide_container': progress_tracker_obj.hide,
        'show_for_operation': progress_tracker_obj.show,
        'update_overall': progress_tracker_obj.update_overall,
        'update_step': progress_tracker_obj.update_step,
        'update_current': progress_tracker_obj.update_current,
        'update_progress': progress_tracker_obj.update,
        'complete_operation': progress_tracker_obj.complete,
        'error_operation': progress_tracker_obj.error,
        'reset_all': progress_tracker_obj.reset
    }
    
    # Help panel
    help_content = """
    <div style="padding: 8px; background: #ffffff;">
        <p style="margin: 6px 0; font-size: 13px;">Proses ini akan mengunduh dan menyinkronkan model pre-trained yang diperlukan.</p>
        <div style="margin: 8px 0;">
            <strong style="color: #495057; font-size: 13px;">Tahapan Proses:</strong>
            <ul style="margin: 4px 0; padding-left: 18px; color: #495057; font-size: 12px;">
                <li><strong>Check:</strong> Memeriksa model yang sudah ada</li>
                <li><strong>Download:</strong> Mengunduh model yang belum ada</li>
                <li><strong>Sync:</strong> Sinkronisasi ke Google Drive</li>
                <li><strong>Verify:</strong> Validasi integritas file</li>
            </ul>
        </div>
        <div style="margin-top: 8px; padding: 6px; background: #e7f3ff; border-radius: 3px; font-size: 12px;">
            <strong>💡 Tips:</strong> Proses akan otomatis skip model yang sudah ada dan valid.
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Info Model Pre-trained")
    help_panel.selected_index = None
    
    # Section headers
    action_header = widgets.HTML(f"""
    <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 10px 0; font-size: 16px; 
               border-bottom: 2px solid {get_color('success', '#28a745')}; padding-bottom: 6px;'>
        {get_icon('play', '▶️')} Actions
    </h4>
    """)
    
    # Main UI assembly
    ui = widgets.VBox([
        header, status_panel, info_html,
        create_divider(), action_header, action_buttons['container'],
        progress_tracker['container'], log_components['log_accordion'], 
        create_divider(), help_panel
    ], layout=widgets.Layout(width='100%', padding='8px', overflow='hidden'))
    
    # Component mapping
    return {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'info_html': info_html,
        'action_buttons': action_buttons,
        'download_sync_button': action_buttons['download_button'],
        'reset_ui_button': action_buttons['check_button'],
        'progress_tracker': progress_tracker,
        'progress_container': progress_tracker['container'],
        'show_for_operation': progress_tracker['show_for_operation'],
        'update_progress': progress_tracker['update_progress'],
        'complete_operation': progress_tracker['complete_operation'],
        'error_operation': progress_tracker['error_operation'],
        'reset_all': progress_tracker['reset_all'],
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],
        'models_dir': models_dir,
        'drive_models_dir': drive_models_dir,
        'auto_check_enabled': True,
        'help_panel': help_panel,
        'module_name': 'pretrained_model'
    }