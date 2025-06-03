"""
File: smartcash/ui/pretrained_model/components/ui_components.py
Deskripsi: UI components pretrained model yang terintegrasi tanpa duplikasi utils
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.pretrained_model.constants.model_constants import MODEL_CONFIGS, DEFAULT_MODELS_DIR, DEFAULT_DRIVE_MODELS_DIR
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create pretrained model UI dengan integrated components"""
    
    get_icon = lambda key, fallback="🧠": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('model', '🧠')} Persiapan Model Pre-trained", 
        "Download dan sinkronisasi model YOLOv5 dan EfficientNet-B4 untuk SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel("Mempersiapkan auto-check model...", "info")
    
    # Model info display menggunakan constants
    models_dir = config.get('models_dir', DEFAULT_MODELS_DIR) if config else DEFAULT_MODELS_DIR
    drive_models_dir = config.get('drive_models_dir', DEFAULT_DRIVE_MODELS_DIR) if config else DEFAULT_DRIVE_MODELS_DIR
    
    # Generate model list dari constants
    model_list_html = ''.join([
        f"<li><b>{cfg['name']}</b> ({cfg['min_size_mb']} MB) - {cfg['description']}</li>"
        for cfg in MODEL_CONFIGS.values()
    ])
    
    info_html = widgets.HTML(f"""
        <div style='padding:10px; background-color:{get_color('alert_info_bg', '#d1ecf1')}; 
                   color:{get_color('alert_info_text', '#0c5460')}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {get_color('alert_info_text', '#0c5460')}'>
            <p style='margin:5px 0'>{get_icon('info', 'ℹ️')} <b>Model yang akan diunduh:</b></p>
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
    
    # Progress tracking
    progress_components = create_progress_tracking_container()
    
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
        <h4 style='color: {get_color('dark', '#333')}; margin: 12px 0 8px 0; font-size: 16px;'>
            {get_icon('download', '📥')} Aksi Model
        </h4>
    """)
    
    # Main UI assembly
    ui = widgets.VBox([
        header, status_panel, info_html,
        create_divider(), action_header, action_buttons['container'],
        progress_components['container'], log_components['log_accordion'], 
        create_divider(), help_panel
    ], layout=widgets.Layout(width='100%', padding='8px', overflow='hidden'))
    
    # Compile components
    ui_components = {
        # Main UI
        'ui': ui, 'header': header, 'status_panel': status_panel,
        'info_html': info_html,
        
        # Action buttons - Map ke nama yang diharapkan handlers
        'action_buttons': action_buttons,
        'download_sync_button': action_buttons['download_button'],  # Primary button maps to download_sync
        'reset_ui_button': action_buttons['check_button'],  # Secondary button maps to reset
        
        # Progress components
        'progress_components': progress_components,
        'progress_container': progress_components['container'],
        'show_for_operation': progress_components.get('show_for_operation'),
        'update_progress': progress_components.get('update_progress'),
        'complete_operation': progress_components.get('complete_operation'),
        'error_operation': progress_components.get('error_operation'),
        'reset_all': progress_components.get('reset_all'),
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],  # Alias untuk kompatibilitas
        
        # Model paths dari constants
        'models_dir': models_dir,
        'drive_models_dir': drive_models_dir,
        
        # Auto-check flag untuk trigger setelah render
        'auto_check_enabled': True,
        
        # UI info
        'help_panel': help_panel,
        'module_name': 'pretrained_model'
    }
    
    # Validate critical components - create fallback buttons if missing
    critical_components = ['download_sync_button', 'reset_ui_button']
    for comp_name in critical_components:
        if ui_components.get(comp_name) is None:
            ui_components[comp_name] = widgets.Button(
                description=comp_name.replace('_', ' ').title(),
                button_style='primary' if 'download' in comp_name else 'info',
                disabled=True,
                tooltip=f"Component {comp_name} tidak tersedia",
                layout=widgets.Layout(width='auto', max_width='150px')
            )
    
    return ui_components