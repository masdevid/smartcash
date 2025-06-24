"""
File: smartcash/ui/setup/env_config/ui_components.py
Deskripsi: UI components untuk environment configuration dengan shared components dan dual-column layout
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.header import create_header
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

def create_env_config_ui() -> Dict[str, Any]:
    """🚀 Membuat complete environment config UI dengan dual-column layout"""
    
    # Header menggunakan shared component
    header = create_header(
        title="Environment Configuration", 
        subtitle="Setup lingkungan SmartCash untuk deteksi mata uang",
        icon="🚀"
    )
    
    # Setup button dengan styling konsisten
    setup_button = widgets.Button(
        description="🚀 Setup Environment",
        button_style='primary',
        layout=widgets.Layout(
            width='220px', 
            height='45px',
            margin='15px auto'
        )
    )
    
    # Button container untuk centering
    setup_button_container = widgets.HBox(
        [setup_button],
        layout=widgets.Layout(
            width='100%',
            justify_content='center',
            margin='15px 0'
        )
    )
    
    # Status panel dengan responsive styling
    status_panel = widgets.HTML(
        value="<p style='color: #007bff; padding: 10px; margin: 5px 0;'>🔍 Memeriksa status environment...</p>",
        layout=widgets.Layout(
            margin='10px 0',
            padding='10px',
            border='1px solid #e9ecef',
            border_radius='8px'
        )
    )
    
    # Progress components dengan enhanced styling
    progress_bar = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100,
        layout=widgets.Layout(
            width='100%',
            margin='5px 0'
        )
    )
    
    progress_text = widgets.HTML(
        value="<span style='color: #6c757d; font-size: 14px;'>Siap untuk setup environment</span>",
        layout=widgets.Layout(margin='5px 0')
    )
    
    # Log accordion menggunakan shared component dengan auto-open
    log_components = create_log_accordion(
        module_name='env_setup', 
        height='200px',
        width='100%'
    )
    log_output = log_components['log_output']
    log_accordion = log_components['log_accordion'] 
    log_accordion.selected_index = 0  # Auto-open accordion
    
    # ===== DUAL COLUMN LAYOUT: Summary (Kiri) + Sistem Colab (Kanan) =====
    
    # Kolom Kiri: Environment Summary
    left_summary_panel = widgets.HTML(
        value="""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #495057;">📋 Environment Summary</h4>
            <div id="env-summary-content">
                <p style="color: #6c757d;">Loading environment information...</p>
            </div>
        </div>
        """,
        layout=widgets.Layout(
            width='48%',
            margin='0 1% 0 0'
        )
    )
    
    # Kolom Kanan: Sistem Colab Info
    right_colab_panel = widgets.HTML(
        value="""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #1976d2;">📊 Informasi Sistem Colab</h4>
            <div id="colab-system-content">
                <p style="color: #1565c0;">Loading sistem Colab information...</p>
            </div>
        </div>
        """,
        layout=widgets.Layout(
            width='48%',
            margin='0 0 0 1%'
        )
    )
    
    # Dual column container
    dual_column_summary = widgets.HBox(
        [left_summary_panel, right_colab_panel],
        layout=widgets.Layout(
            width='100%',
            margin='15px 0',
            justify_content='space-between'
        )
    )
    
    # Tips panel dengan enhanced styling
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
    
    # Main layout assembly dengan flexbox
    main_layout = widgets.VBox([
        header,
        setup_button_container,
        status_panel,
        widgets.VBox([
            progress_bar, 
            progress_text
        ], layout=widgets.Layout(margin='10px 0')),
        log_accordion,
        dual_column_summary,  # ✨ Dual column layout
        tips_panel
    ], layout=widgets.Layout(
        width='100%',
        padding='20px',
        box_sizing='border-box'
    ))
    
    # Component dictionary dengan semua references
    components = {
        'ui': main_layout,
        'header': header,
        'setup_button': setup_button,
        'setup_button_container': setup_button_container,
        'status_panel': status_panel,
        'progress_bar': progress_bar,
        'progress_text': progress_text,
        'log_accordion': log_accordion,
        'log_output': log_output,
        'left_summary_panel': left_summary_panel,    # ✨ Kolom kiri
        'right_colab_panel': right_colab_panel,      # ✨ Kolom kanan  
        'dual_column_summary': dual_column_summary,  # ✨ Container dual column
        'tips_panel': tips_panel
    }
    
    return components

def setup_ui_logger_bridge(ui_components: Dict[str, Any], namespace: str = "ENV") -> Any:
    """🔧 Setup logger bridge untuk UI components dengan namespace"""
    logger_bridge = create_ui_logger_bridge(ui_components, f"smartcash.ui.setup.env_config")
    ui_components['logger'] = logger_bridge
    ui_components['logger_namespace'] = namespace
    return logger_bridge

def update_summary_panels(ui_components: Dict[str, Any], 
                         env_summary: str = None, 
                         colab_info: str = None) -> None:
    """🔄 Update dual summary panels dengan informasi terbaru"""
    if env_summary and 'left_summary_panel' in ui_components:
        # Update kolom kiri - Environment Summary
        env_html = f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #495057;">📋 Environment Summary</h4>
            <div id="env-summary-content">
                {env_summary}
            </div>
        </div>
        """
        ui_components['left_summary_panel'].value = env_html
    
    if colab_info and 'right_colab_panel' in ui_components:
        # Update kolom kanan - Sistem Colab Info
        colab_html = f"""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto;">
            <h4 style="margin-top: 0; color: #1976d2;">📊 Informasi Sistem Colab</h4>
            <div id="colab-system-content">
                {colab_info}
            </div>
        </div>
        """
        ui_components['right_colab_panel'].value = colab_html