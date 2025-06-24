"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Shared UI components untuk environment configuration
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional
from smartcash.ui.setup.env_config.constants import UI_ELEMENTS, STATUS_COLORS

def create_env_config_component(setup_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🎨 Create complete environment config UI component"""
    
    # Header component
    header = _create_header_component()
    
    # Setup button (centered)
    setup_button = _create_setup_button(setup_callback)
    
    # Status panel
    status_panel = _create_status_panel()
    
    # Progress tracker
    progress_bar, progress_text = _create_progress_components()
    
    # Log accordion
    log_accordion = _create_log_accordion()
    
    # Environment summary
    summary_panel = _create_summary_panel()
    
    # Tips & Requirements (flexbox layout)
    tips_panel = _create_tips_panel()
    
    # Main layout dengan flexbox
    layout = _create_main_layout([
        header,
        setup_button,
        status_panel,
        widgets.VBox([progress_bar, progress_text]),
        log_accordion,
        summary_panel,
        tips_panel
    ])
    
    # Component dictionary
    components = {
        UI_ELEMENTS['setup_button']: setup_button,
        UI_ELEMENTS['status_panel']: status_panel,
        UI_ELEMENTS['progress_bar']: progress_bar,
        UI_ELEMENTS['progress_text']: progress_text,
        UI_ELEMENTS['log_accordion']: log_accordion,
        UI_ELEMENTS['summary_panel']: summary_panel,
        'header': header,
        'tips_panel': tips_panel,
        'main_layout': layout
    }
    
    return components

def _create_header_component() -> widgets.HTML:
    """📌 Create header component"""
    return widgets.HTML(
        value="""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 15px;">
            <h2 style="margin: 0; font-size: 24px;">🚀 SmartCash Environment Setup</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Konfigurasi environment untuk YOLOv5 + EfficientNet-B4</p>
        </div>
        """
    )

def _create_setup_button(callback: Optional[Callable] = None) -> widgets.Button:
    """🔘 Create setup button"""
    button = widgets.Button(
        description='Setup Environment',
        button_style='success',
        layout=widgets.Layout(
            width='300px',
            height='45px',
            margin='10px auto',
            display='flex'
        ),
        style={'font_weight': 'bold'}
    )
    
    if callback:
        button.on_click(lambda b: callback())
    
    return button

def _create_status_panel() -> widgets.HTML:
    """📊 Create status panel"""
    return widgets.HTML(
        value=_get_initial_status_html(),
        layout=widgets.Layout(margin='10px 0')
    )

def _create_progress_components() -> tuple:
    """📈 Create progress bar dan text"""
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        bar_style='info',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    progress_text = widgets.HTML(
        value='<div style="text-align: center; color: #666;">Siap untuk setup...</div>',
        layout=widgets.Layout(margin='5px 0')
    )
    
    return progress_bar, progress_text

def _create_log_accordion() -> widgets.Accordion:
    """📝 Create collapsible log accordion"""
    log_content = widgets.HTML(
        value='<div style="font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; padding: 10px; background: #f8f9fa; border-radius: 5px;">Log akan muncul di sini...</div>'
    )
    
    accordion = widgets.Accordion(children=[log_content])
    accordion.set_title(0, '📝 Setup Logs')
    accordion.selected_index = None  # Collapsed by default
    
    return accordion

def _create_summary_panel() -> widgets.HTML:
    """📋 Create environment summary panel"""
    return widgets.HTML(
        value=_get_initial_summary_html(),
        layout=widgets.Layout(margin='15px 0')
    )

def _create_tips_panel() -> widgets.HTML:
    """💡 Create tips & requirements panel dengan flexbox layout"""
    return widgets.HTML(
        value="""
        <div style="display: flex; gap: 15px; width: 100%; margin: 15px 0;">
            <div style="flex: 1; padding: 15px; background: #e3f2fd; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: #1976d2; line-height: 1.2;">💡 Tips Setup</h4>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.3;">
                    <li style="margin: 0; padding: 0;">Pastikan Google Drive sudah ter-mount</li>
                    <li style="margin: 0; padding: 0;">Proses setup memakan waktu 2-3 menit</li>
                    <li style="margin: 0; padding: 0;">Jangan tutup browser saat setup berlangsung</li>
                    <li style="margin: 0; padding: 0;">Setup hanya perlu dilakukan sekali</li>
                </ul>
            </div>
            <div style="flex: 1; padding: 15px; background: #f3e5f5; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: #7b1fa2; line-height: 1.2;">📋 Requirements</h4>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.3;">
                    <li style="margin: 0; padding: 0;">Google Colab environment</li>
                    <li style="margin: 0; padding: 0;">Google Drive access permission</li>
                    <li style="margin: 0; padding: 0;">Minimal 2GB free space di Drive</li>
                    <li style="margin: 0; padding: 0;">Koneksi internet stabil</li>
                </ul>
            </div>
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )

def _create_main_layout(components: list) -> widgets.VBox:
    """📐 Create main layout container dengan flexbox styling"""
    return widgets.VBox(
        components,
        layout=widgets.Layout(
            width='100%',
            padding='20px',
            border='1px solid #ddd',
            border_radius='10px',
            box_shadow='0 2px 10px rgba(0,0,0,0.1)',
            display='flex',
            flex_flow='column',
            gap='10px'
        )
    )

def _get_initial_status_html() -> str:
    """Get initial status HTML"""
    return f"""
    <div style="padding: 15px; border-left: 4px solid {STATUS_COLORS['info']}; background: rgba(33, 150, 243, 0.1); border-radius: 5px;">
        <strong>🔍 Mengecek status environment...</strong>
        <div style="margin-top: 5px; color: #666; font-size: 14px;">
            Sedang menganalisis konfigurasi environment SmartCash
        </div>
    </div>
    """

def _get_initial_summary_html() -> str:
    """Get initial summary HTML"""
    return """
    <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
        <h4 style="margin-top: 0; color: #495057;">📊 Environment Summary</h4>
        <div style="color: #6c757d;">
            Summary akan ditampilkan setelah analisis environment selesai...
        </div>
    </div>
    """

# Utility functions untuk update components
def update_component_status(components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """Update status component"""
    from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel
    update_status_panel(components, message, status_type)

def update_component_progress(components: Dict[str, Any], progress: int, message: str = "", is_error: bool = False) -> None:
    """Update progress components"""
    from smartcash.ui.setup.env_config.utils.ui_updater import update_progress_bar
    update_progress_bar(components, progress, message, is_error)

def update_component_button(components: Dict[str, Any], enabled: bool = True, text: str = "Setup Environment") -> None:
    """Update setup button"""
    from smartcash.ui.setup.env_config.utils.ui_updater import update_setup_button
    update_setup_button(components, enabled, text)

def append_component_log(components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """Append to log component"""
    from smartcash.ui.setup.env_config.utils.ui_updater import append_to_log
    append_to_log(components, message, level)