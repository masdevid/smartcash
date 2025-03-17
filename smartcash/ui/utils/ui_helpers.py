"""
File: smartcash/ui/utils/ui_helpers.py
Deskripsi: Fungsi helper untuk komponen UI dan integrasi dengan notebook dengan styling yang lebih baik
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Callable, Any, Optional, Dict, List, Union, Tuple
import time
from pathlib import Path
import datetime

from smartcash.ui.utils.constants import COLORS, ICONS, FILE_SIZE_UNITS
from smartcash.ui.components.widget_layouts import (
    BUTTON_LAYOUTS, GROUP_LAYOUTS, CONTAINER_LAYOUTS, CONTENT_LAYOUTS
)

def set_active_theme(theme_name: str = 'default') -> bool:
    """
    Set tema aktif untuk UI komponen.
    
    Args:
        theme_name: Nama tema ('default' atau 'dark')
        
    Returns:
        Boolean menunjukkan keberhasilan operasi
    """
    from smartcash.ui.utils.constants import THEMES
    
    global ACTIVE_THEME
    if theme_name in THEMES:
        ACTIVE_THEME = theme_name
        return True
    return False

def inject_css_styles() -> None:
    """Inject CSS styles untuk komponen UI."""
    from smartcash.ui.utils.constants import COLORS, FONTS, SIZES
    
    css = """
    <style>
    .smartcash-header {
        background-color: %s;
        padding: 15px;
        color: black;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 5px solid %s;
    }
    
    .smartcash-section {
        color: %s;
        font-size: %s; 
        margin-top: 15px;
        margin-bottom: 10px;
        font-family: %s;
    }
    
    .smartcash-info-box {
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
        border: 1px solid %s;
    }
    
    .smartcash-status {
        margin: 5px 0;
        padding: 8px 12px;
        border-radius: 4px;
    }
    
    /* Alert Styles */
    .smartcash-alert-info {
        background-color: %s;
        color: %s;
        border-left: 4px solid %s;
    }
    
    .smartcash-alert-success {
        background-color: %s;
        color: %s;
        border-left: 4px solid %s;
    }
    
    .smartcash-alert-warning {
        background-color: %s;
        color: %s;
        border-left: 4px solid %s;
    }
    
    .smartcash-alert-error {
        background-color: %s;
        color: %s;
        border-left: 4px solid %s;
    }

    /* Button Styles */
    .jupyter-button {
        font-family: %s;
        font-size: %s;
        transition: all 0.2s ease;
    }
    
    .jupyter-button:hover {
        opacity: 0.9;
    }
    
    /* Progress bar styles */
    .widget-progressbar {
        height: 20px;
        border-radius: 4px;
    }
    
    .widget-hbox, .widget-vbox {
        overflow: visible;
    }
    </style>
    """ % (
        COLORS['header_bg'],
        COLORS['header_border'],
        COLORS['dark'],
        SIZES['lg'],
        FONTS['header'],
        COLORS['border'],
        COLORS['alert_info_bg'],
        COLORS['alert_info_text'],
        COLORS['alert_info_text'],
        COLORS['alert_success_bg'],
        COLORS['alert_success_text'],
        COLORS['alert_success_text'],
        COLORS['alert_warning_bg'],
        COLORS['alert_warning_text'],
        COLORS['alert_warning_text'],
        COLORS['alert_danger_bg'],
        COLORS['alert_danger_text'],
        COLORS['alert_danger_text'],
        FONTS['default'],
        SIZES['md']
    )
    
    display(HTML(css))

def format_file_size(size_bytes: int) -> str:
    """
    Format ukuran file menjadi string yang mudah dibaca.
    
    Args:
        size_bytes: Ukuran file dalam bytes
        
    Returns:
        String berisi ukuran file dengan unit yang sesuai
    """
    if size_bytes == 0:
        return "0B"
    
    i = 0
    while size_bytes >= 1024 and i < len(FILE_SIZE_UNITS) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f}{FILE_SIZE_UNITS[i]}"

def run_task(task_func, with_output=None):
    """
    Eksekusi fungsi tugas dan tangani error.
    
    Args:
        task_func: Fungsi task yang akan dijalankan
        with_output: Widget output untuk menampilkan error
        
    Returns:
        Hasil dari task_func atau None jika terjadi error
    """
    try:
        # Run task langsung
        return task_func()
    except Exception as e:
        # Handle error
        if with_output:
            from smartcash.ui.components.alerts import create_status_indicator
            with with_output:
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        # Re-raise exception untuk penanganan lebih lanjut jika diperlukan
        raise

def create_confirmation_dialog(title, message, on_confirm, on_cancel=None, 
                             confirm_label="Konfirmasi", cancel_label="Batal"):
    """
    Buat dialog konfirmasi.
    
    Args:
        title: Judul dialog
        message: Pesan dialog
        on_confirm: Callback saat dikonfirmasi
        on_cancel: Callback saat dibatalkan
        confirm_label: Label tombol konfirmasi
        cancel_label: Label tombol batal
        
    Returns:
        Widget VBox berisi dialog konfirmasi
    """
    # Dialog content
    content = widgets.VBox([
        widgets.HTML(f"""
        <div style="padding:15px; background-color:{COLORS['alert_warning_bg']}; 
                     color:{COLORS['alert_warning_text']}; 
                     border-left:4px solid {COLORS['alert_warning_text']}; 
                     border-radius:4px; margin:10px 0;">
            <h4 style="margin-top:0; font-family:{FONTS['header']};">{ICONS['warning']} {title}</h4>
            <p style="margin-bottom:0;">{message}</p>
        </div>
        """),
        widgets.HBox([
            widgets.Button(
                description=cancel_label,
                button_style="warning",
                icon='times',
                layout=BUTTON_LAYOUTS['small'],
                tooltip="Batalkan operasi"
            ),
            widgets.Button(
                description=confirm_label,
                button_style="danger",
                icon='check',
                layout=BUTTON_LAYOUTS['small'],
                tooltip="Konfirmasi operasi"
            )
        ], layout=GROUP_LAYOUTS['horizontal'])
    ], layout=CONTAINER_LAYOUTS['card'])
    
    # Set callbacks
    content.children[1].children[0].on_click(
        lambda b: on_cancel() if on_cancel else None
    )
    content.children[1].children[1].on_click(
        lambda b: on_confirm()
    )
    
    return content

def create_button_group(buttons, layout=None):
    """
    Buat group tombol dengan layout konsisten.
    
    Args:
        buttons: List of tuples (label, style, icon, callback)
        layout: Layout opsional untuk container
        
    Returns:
        Widget HBox berisi tombol-tombol
    """
    from smartcash.ui.utils.constants import BUTTON_STYLES
    
    btn_widgets = []
    
    for label, style, icon, callback in buttons:
        btn = widgets.Button(
            description=label,
            button_style=BUTTON_STYLES.get(style, ''),
            icon=icon,
            layout=BUTTON_LAYOUTS['small'],
            tooltip=label
        )
        
        if callback:
            btn.on_click(callback)
            
        btn_widgets.append(btn)
    
    return widgets.HBox(
        btn_widgets, 
        layout=layout or GROUP_LAYOUTS['horizontal']
    )

def create_loading_indicator(message: str = "Memproses...") -> Tuple[widgets.HBox, Callable]:
    """
    Buat indikator loading dengan callback untuk show/hide.
    
    Args:
        message: Pesan default untuk ditampilkan
        
    Returns:
        Tuple berisi (widget loading, fungsi toggle)
    """
    spinner = widgets.HTML(
        value=f'<i class="fa fa-spinner fa-spin" style="font-size: 20px; color: {COLORS["primary"]};"></i>'
    )
    
    message_widget = widgets.HTML(value=f'<span style="margin-left: 10px;">{message}</span>')
    
    loading = widgets.HBox(
        [spinner, message_widget],
        layout=widgets.Layout(
            display='none',
            align_items='center',
            margin='10px 0',
            width='auto'
        )
    )
    
    def toggle_loading(show: bool = True, new_message: Optional[str] = None):
        loading.layout.display = 'flex' if show else 'none'
        if new_message is not None:
            message_widget.value = f'<span style="margin-left: 10px;">{new_message}</span>'
    
    return loading, toggle_loading

def create_progress_updater(progress_bar: widgets.IntProgress) -> Callable:
    """
    Buat fungsi updater untuk progress bar.
    
    Args:
        progress_bar: Widget IntProgress untuk diupdate
        
    Returns:
        Fungsi updater untuk progress bar
    """
    def update_progress(value: int, total: int, message: Optional[str] = None):
        # Update value
        progress_bar.max = total
        progress_bar.value = value
        
        # Update description jika ada message
        if message:
            progress_bar.description = message
        else:
            progress_pct = min(100, int(value * 100 / total) if total > 0 else 0)
            progress_bar.description = f"{progress_pct}%"
    
    return update_progress

def display_file_info(file_path: str, description: Optional[str] = None) -> widgets.HTML:
    """
    Tampilkan informasi file.
    
    Args:
        file_path: Path ke file
        description: Deskripsi opsional
        
    Returns:
        Widget HTML berisi info file
    """
    # Ambil info file
    path = Path(file_path)
    if path.exists():
        file_size = path.stat().st_size
        file_time = path.stat().st_mtime
        
        # Format ukuran file
        formatted_size = format_file_size(file_size)
        
        # Format waktu
        time_str = datetime.datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Buat HTML
        html = f"""
        <div style="padding: 15px; background-color: {COLORS['light']}; 
                  border-radius: 5px; margin: 10px 0; 
                  border: 1px solid {COLORS['border']};">
            <p><strong>{ICONS['file']} File:</strong> {path.name}</p>
            <p><strong>{ICONS['folder']} Path:</strong> {path.parent}</p>
            <p><strong>📏 Size:</strong> {formatted_size}</p>
            <p><strong>{ICONS['time']} Modified:</strong> {time_str}</p>
        """
        
        if description:
            html += f"<p><strong>📝 Description:</strong> {description}</p>"
        
        html += "</div>"
        
        return widgets.HTML(value=html)
    else:
        return widgets.HTML(value=f"""
        <div style="padding: 10px; background-color: {COLORS['alert_warning_bg']}; 
                  color: {COLORS['alert_warning_text']}; border-radius: 5px; margin: 10px 0;
                  border-left: 4px solid {COLORS['alert_warning_text']};">
            <p>{ICONS['warning']} File tidak ditemukan: {file_path}</p>
        </div>
        """)

def update_output_area(output_widget: widgets.Output, message: str, status: str = 'info', clear: bool = False):
    """
    Update area output dengan status baru.
    
    Args:
        output_widget: Widget output untuk diupdate
        message: Pesan untuk ditampilkan
        status: Jenis status ('info', 'success', 'warning', 'error')
        clear: Apakah perlu clear output sebelumnya
    """
    from smartcash.ui.components.alerts import create_status_indicator
    
    with output_widget:
        if clear:
            clear_output(wait=True)
        display(create_status_indicator(status, message))

def register_observer_callback(observer_manager, event_type, output_widget, 
                            group_name="ui_observer_group"):
    """
    Register callback untuk observer events.
    
    Args:
        observer_manager: ObserverManager instance
        event_type: Tipe event untuk di-observe
        output_widget: Widget output untuk menampilkan updates
        group_name: Nama group untuk observer
    """
    if observer_manager:
        from smartcash.ui.components.alerts import create_status_indicator
        
        def update_ui_callback(event_type, sender, message=None, **kwargs):
            if message:
                status = kwargs.get('status', 'info')
                with output_widget:
                    display(create_status_indicator(status, message))
        
        # Register observer
        observer_manager.create_simple_observer(
            event_type=event_type,
            callback=update_ui_callback,
            name=f"UI_{event_type}_Observer",
            group=group_name
        )

def create_divider() -> widgets.HTML:
    """Buat divider horizontal."""
    return widgets.HTML(f"<hr style='margin: 15px 0; border: 0; border-top: 1px solid {COLORS['border']};'>")

def create_spacing(height: str = '10px') -> widgets.HTML:
    """Buat elemen spacing untuk mengatur jarak antar komponen."""
    return widgets.HTML(f"<div style='height: {height};'></div>")