# File: smartcash/ui/setup/env_config/components/ui_components.py
# Deskripsi: Komponen UI untuk environment configuration - HANYA header dan logger menggunakan shared components

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.header import create_header
from smartcash.ui.components.log_accordion import create_log_accordion

def create_env_config_ui() -> Dict[str, Any]:
    """🎛️ Create complete environment configuration UI - HANYA header dan logger diubah"""
    
    # FIXED: Gunakan shared header component
    header = create_header(
        title="Environment Configuration",
        description="Setup lingkungan SmartCash untuk deteksi mata uang YOLOv5 + EfficientNet-B4",
        icon="🔧"
    )
    
    # Setup button dengan styling - TIDAK DIUBAH
    setup_button = widgets.Button(
        description="🚀 Setup Environment",
        button_style='primary',
        layout=widgets.Layout(width='220px', height='45px')
    )
    
    # Status panel untuk summary environment - TIDAK DIUBAH
    status_panel = widgets.HTML(
        value="<p style='color: #007bff; padding: 10px;'>🔍 Memeriksa status environment...</p>",
        layout=widgets.Layout(
            margin='10px 0',
            padding='10px',
            border='1px solid #e9ecef',
            border_radius='8px'
        )
    )
    
    # Progress components - TIDAK DIUBAH
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    progress_text = widgets.HTML(
        value="<span style='color: #6c757d; font-size: 14px;'>Siap untuk setup environment</span>",
        layout=widgets.Layout(margin='5px 0')
    )
    
    # FIXED: Gunakan shared log_accordion component
    log_components = create_log_accordion(
        module_name='Environment Config',
        height='250px',
        width='100%'
    )
    log_output = log_components['log_output']
    log_accordion = log_components['log_accordion']
    
    # Dual column summary panels - TIDAK DIUBAH
    left_summary_panel = widgets.HTML(
        value="""
        <h4 style="color: #2c3e50; margin-bottom: 10px; font-size: 16px;">
            🖥️ Environment Information
        </h4>
        <div style="color: #555; font-size: 13px; line-height: 1.4;">
            <div style="margin: 8px 0;"><strong>Platform:</strong> <span id="platform-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>Python:</strong> <span id="python-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>PyTorch:</strong> <span id="torch-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>CUDA:</strong> <span id="cuda-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>Memory:</strong> <span id="memory-info">Memuat...</span></div>
        </div>
        """,
        layout=widgets.Layout(
            width='48%', 
            padding='15px', 
            border='1px solid #e9ecef', 
            border_radius='8px', 
            margin='10px 1% 10px 0'
        )
    )
    
    right_colab_panel = widgets.HTML(
        value="""
        <h4 style="color: #2c3e50; margin-bottom: 10px; font-size: 16px;">
            ☁️ Google Colab Information
        </h4>
        <div style="color: #555; font-size: 13px; line-height: 1.4;">
            <div style="margin: 8px 0;"><strong>Runtime:</strong> <span id="runtime-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>GPU:</strong> <span id="gpu-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>Drive:</strong> <span id="drive-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>Storage:</strong> <span id="storage-info">Memuat...</span></div>
            <div style="margin: 8px 0;"><strong>Session:</strong> <span id="session-info">Memuat...</span></div>
        </div>
        """,
        layout=widgets.Layout(
            width='48%', 
            padding='15px', 
            border='1px solid #e9ecef', 
            border_radius='8px', 
            margin='10px 0 10px 1%'
        )
    )
    
    dual_column_summary = widgets.HBox([
        left_summary_panel, 
        right_colab_panel
    ], layout=widgets.Layout(width='100%', margin='20px 0'))
    
    # Tips panel - TIDAK DIUBAH
    tips_html = """
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                padding: 20px; border-radius: 12px; border-left: 4px solid #2196f3;">
        <h4 style="margin-top: 0; color: #1976d2; display: flex; align-items: center;">
            💡 Tips & Requirements
        </h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <ul style="margin: 0; padding-left: 20px; color: #424242;">
                    <li>Pastikan Google Drive memiliki ruang minimal 12GB</li>
                    <li>Setup akan membuat folder struktur data lengkap</li>
                </ul>
            </div>
            <div>
                <ul style="margin: 0; padding-left: 20px; color: #424242;">
                    <li>Proses setup memerlukan waktu 1-2 menit</li>
                    <li>Koneksi internet stabil diperlukan</li>
                </ul>
            </div>
        </div>
    </div>
    """
    tips_panel = widgets.HTML(
        value=tips_html,
        layout=widgets.Layout(margin='20px 0')
    )
    
    # Main layout - TIDAK DIUBAH
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
    
    # UI components dictionary - TIDAK DIUBAH
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
    """🔧 Setup logger bridge untuk UI components - TIDAK DIUBAH"""
    from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
    logger_bridge = create_ui_logger_bridge(ui_components, f"smartcash.ui.setup.env_config.{namespace}")
    return logger_bridge