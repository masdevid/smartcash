"""
File: smartcash/ui/setup/env_config/components/ui_components.py
Deskripsi: UI components yang diperbaiki untuk menampilkan status environment dengan benar
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, Optional

def create_environment_summary_panel() -> widgets.VBox:
    """🌍 Environment summary panel dengan status yang akurat"""
    
    # Header dengan emoji
    header = widgets.HTML(
        value="<h4>🌍 Environment Summary</h4>",
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Status indicators (akan diupdate via handler)
    status_html = widgets.HTML(
        value="""
        <div class="status-summary">
            <div class="status-item">
                <span class="status-label">Status:</span>
                <span class="status-value" id="overall-status">🔍 Checking...</span>
            </div>
            <div class="status-item">  
                <span class="status-label">Python Environment:</span>
                <span class="status-value" id="python-status">🔍 Checking...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Google Drive:</span>
                <span class="status-value" id="drive-status">🔍 Checking...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Configurations:</span>
                <span class="status-value" id="config-status">🔍 Checking...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Directory Structure:</span>
                <span class="status-value" id="directory-status">🔍 Checking...</span>
            </div>
        </div>
        <style>
        .status-summary {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            padding: 2px 0;
        }
        .status-label {
            font-weight: 500;
            color: #495057;
        }
        .status-value {
            font-weight: 600;
        }
        </style>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    # Drive mount info (terpisah dari status)
    drive_info = widgets.HTML(
        value="",  # Akan diupdate via handler
        layout=widgets.Layout(margin='10px 0 0 0')
    )
    
    return widgets.VBox([
        header,
        status_html,
        drive_info
    ], layout=widgets.Layout(
        border='1px solid #dee2e6',
        border_radius='8px',
        padding='15px',
        margin='10px 0'
    ))

def create_setup_control_panel() -> widgets.VBox:
    """⚙️ Setup control panel dengan progress tracking"""
    
    # Header
    header = widgets.HTML(
        value="<h4>⚙️ Environment Setup</h4>",
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Setup button dengan styling
    setup_button = widgets.Button(
        description='🚀 Setup Environment',
        button_style='primary',
        layout=widgets.Layout(width='200px', height='40px'),
        style={'font_weight': 'bold'}
    )
    
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Progress text
    progress_text = widgets.HTML(
        value="<div style='text-align: center; color: #6c757d;'>Klik tombol setup untuk memulai</div>",
        layout=widgets.Layout(margin='5px 0')
    )
    
    # Log output dengan accordion
    log_output = widgets.Output(
        layout=widgets.Layout(height='200px', overflow='auto')
    )
    
    log_accordion = widgets.Accordion(
        children=[log_output],
        titles=('Setup Logs',),
        layout=widgets.Layout(margin='10px 0 0 0')
    )
    log_accordion.selected_index = None  # Collapsed by default
    
    return widgets.VBox([
        header,
        setup_button,
        progress_bar,
        progress_text,
        log_accordion
    ], layout=widgets.Layout(
        border='1px solid #dee2e6',
        border_radius='8px',
        padding='15px',
        margin='10px 0'
    ))

def create_env_config_ui() -> Dict[str, Any]:
    """🎛️ Create complete environment configuration UI menggunakan shared components"""
    
    # Header menggunakan shared component
    header = create_header(
        title="Environment Configuration", 
        description="Setup lingkungan SmartCash untuk deteksi mata uang YOLOv5 + EfficientNet-B4",
        icon="🔧"
    )
    
    # Setup button dengan styling
    setup_button = widgets.Button(
        description="🚀 Setup Environment",
        button_style='primary',
        layout=widgets.Layout(width='220px', height='45px')
    )
    
    # Status panel untuk summary environment
    status_panel = widgets.HTML(
        value="<p style='color: #007bff; padding: 10px;'>🔍 Memeriksa status environment...</p>",
        layout=widgets.Layout(
            margin='10px 0',
            padding='10px',
            border='1px solid #e9ecef',
            border_radius='8px'
        )
    )
    
    # Progress components
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    progress_text = widgets.HTML(
        value="<span style='color: #6c757d; font-size: 14px;'>Siap untuk setup environment</span>",
        layout=widgets.Layout(margin='5px 0')
    )
    
    # Log accordion menggunakan shared component
    log_components = create_log_accordion(
        module_name='env_setup', 
        height='200px',
        width='100%'
    )
    log_output = log_components['log_output']
    log_accordion = log_components['log_accordion'] 
    
    # Environment summary panel (kiri)
    left_summary_panel = widgets.HTML(
        value="""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #495057;">📋 Environment Summary</h4>
            <div id="env-summary-content">
                <p style="color: #6c757d;">Loading environment information...</p>
            </div>
        </div>
        """,
        layout=widgets.Layout(width='48%', margin='0 1% 0 0')
    )
    
    # Sistem Colab info panel (kanan)
    right_colab_panel = widgets.HTML(
        value="""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #1976d2;">📊 Informasi Sistem Colab</h4>
            <div id="colab-system-content">
                <p style="color: #1565c0;">Loading sistem Colab information...</p>
            </div>
        </div>
        """,
        layout=widgets.Layout(width='48%', margin='0 0 0 1%')
    )
    
    # Dual column container
    dual_column_summary = widgets.HBox(
        [left_summary_panel, right_colab_panel],
        layout=widgets.Layout(width='100%', margin='15px 0')
    )
    
    # Tips panel
    tips_panel = widgets.HTML(
        value="""
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin-top: 0; color: #856404;">💡 Tips Setup</h5>
            <ul style="margin: 5px 0; color: #856404;">
                <li>Pastikan Google Drive sudah di-mount sebelum setup</li>
                <li>Setup akan membuat folder SmartCash di Drive Anda</li>
                <li>Template konfigurasi akan disalin otomatis</li>
            </ul>
        </div>
        """
    )
    
    # Main layout
    main_layout = widgets.VBox([
        header,
        widgets.HBox([setup_button], layout=widgets.Layout(justify_content='center', margin='15px 0')),
        status_panel,
        progress_bar,
        progress_text,
        log_accordion,
        dual_column_summary,
        tips_panel
    ], layout=widgets.Layout(width='100%', padding='20px'))
    
    # UI components dictionary
    components = {
        'ui': main_layout,
        'setup_button': setup_button,
        'status_panel': status_panel,
        'progress_bar': progress_bar,
        'progress_text': progress_text,
        'log_accordion': log_accordion,
        'log_output': log_output,
        'left_summary_panel': left_summary_panel,
        'right_colab_panel': right_colab_panel,
        'dual_column_summary': dual_column_summary,
        'tips_panel': tips_panel
    }
    
    return components

def setup_ui_logger_bridge(ui_components: Dict[str, Any], namespace: str = "ENV") -> Any:
    """🔧 Setup logger bridge untuk UI components"""
    logger_bridge = create_ui_logger_bridge(ui_components, f"smartcash.ui.setup.env_config.{namespace}")
    ui_components['logger'] = logger_bridge
    ui_components['logger_namespace'] = namespace
    return logger_bridge

def update_environment_status(ui_components: Dict[str, Any], status_data: Dict[str, Any]) -> None:
    """🔄 Update environment status display dengan data dari status handler"""
    if not status_data or 'summary' not in status_data:
        return
    
    summary = status_data['summary']
    
    # Update environment summary (kiri)
    env_summary_html = f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
        <h4 style="margin-top: 0; color: #495057;">📋 Environment Summary</h4>
        <div class="status-item">
            <strong>Status:</strong> {summary.get('overall_status', '❓ Unknown')}
        </div>
        <div class="status-item">  
            <strong>Python Environment:</strong> {summary.get('python_status', '❓ Unknown')}
        </div>
        <div class="status-item">
            <strong>Google Drive:</strong> {summary.get('drive_status', '❓ Unknown')}
        </div>
        <div class="status-item">
            <strong>Configurations:</strong> {summary.get('config_status', '❓ Unknown')}
        </div>
        <div class="status-item">
            <strong>Directory Structure:</strong> {summary.get('directory_status', '❓ Unknown')}
        </div>
        <hr style="margin: 15px 0;">
        <div style="color: #6c757d; font-size: 14px;">
            {summary.get('setup_message', 'Status check dalam progress...')}
        </div>
    </div>
    """
    
    if 'left_summary_panel' in ui_components:
        ui_components['left_summary_panel'].value = env_summary_html
    
    # Show drive mount info di colab panel (kanan) jika ada
    mount_info = summary.get('mount_info', '')
    if mount_info and 'right_colab_panel' in ui_components:
        colab_html = f"""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #1976d2;">📊 Informasi Sistem Colab</h4>
            <div style="background: #fff; padding: 10px; border-radius: 4px; margin: 10px 0; font-family: monospace;">
                {mount_info}
            </div>
            <div style="color: #1565c0; font-size: 14px; margin-top: 10px;">
                Runtime information dan sistem status akan ditampilkan di sini setelah setup.
            </div>
        </div>
        """
        ui_components['right_colab_panel'].value = colab_html