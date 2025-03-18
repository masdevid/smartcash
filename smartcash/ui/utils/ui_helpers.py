"""
File: smartcash/ui/utils/ui_helpers.py
Deskripsi: Fungsi helper terstandarisasi untuk komponen UI dan integrasi dengan notebook
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import time
from pathlib import Path
import datetime
import re

from smartcash.ui.utils.constants import COLORS, ICONS, ALERT_STYLES

# Pengaturan Tema
def set_active_theme(theme_name: str = 'default') -> bool:
    """
    Set tema aktif untuk UI komponen.
    
    Args:
        theme_name: Nama tema ('default' atau 'dark')
        
    Returns:
        Boolean menunjukkan keberhasilan operasi
    """
    from smartcash.ui.utils.constants import THEMES
    
    if theme_name in THEMES:
        global ACTIVE_THEME
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

# Header Components
def create_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat komponen header dengan style konsisten.
    
    Args:
        title: Judul header
        description: Deskripsi opsional
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi header
    """
    # Tambahkan ikon jika disediakan
    title_with_icon = f"{icon} {title}" if icon else title
    
    header_html = f"""
    <div style="background-color: {COLORS['header_bg']}; padding: 15px; color: black; 
            border-radius: 5px; margin-bottom: 15px; border-left: 5px solid {COLORS['primary']};">
        <h2 style="color: {COLORS['dark']}; margin-top: 0;">{title_with_icon}</h2>
    """
    
    if description:
        header_html += f'<p style="color: {COLORS["dark"]}; margin-bottom: 0;">{description}</p>'
    
    header_html += "</div>"
    
    return widgets.HTML(value=header_html)

def create_section_title(title: str, icon: Optional[str] = "") -> widgets.HTML:
    """
    Buat judul section dengan style konsisten.
    
    Args:
        title: Judul section
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi judul section
    """
    return widgets.HTML(f"""
    <h3 style="color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;">
        {icon} {title}
    </h3>
    """)

# Alert Components
def create_status_indicator(status: str, message: str) -> HTML:
    """
    Buat indikator status dengan style yang sesuai.
    
    Args:
        status: Jenis status ('info', 'success', 'warning', 'error')
        message: Pesan status
        
    Returns:
        HTML berisi indikator status
    """
    style_config = ALERT_STYLES.get(status, ALERT_STYLES['info'])
    
    return HTML(f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: {COLORS['light']};">
        <span style="color: {style_config['text_color']}; font-weight: bold;"> 
            {style_config['icon']} {message}
        </span>
    </div>
    """)

def create_info_alert(message: str, alert_type: str = 'info', icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat alert box dengan style yang sesuai.
    
    Args:
        message: Pesan alert
        alert_type: Jenis alert ('info', 'success', 'warning', 'error')
        icon: Emoji icon opsional, jika tidak diisi akan menggunakan icon default
        
    Returns:
        Widget HTML berisi alert
    """
    style_config = ALERT_STYLES.get(alert_type, ALERT_STYLES['info'])
    icon_str = icon if icon else style_config['icon']
    
    alert_html = f"""
    <div style="padding: 10px; 
                background-color: {style_config['bg_color']}; 
                color: {style_config['text_color']}; 
                border-left: 4px solid {style_config['border_color']}; 
                border-radius: 5px; 
                margin: 10px 0;">
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 10px; font-size: 1.2em;">{icon_str}</div>
            <div>{message}</div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=alert_html)

def create_info_box(title: str, content: str, style: str = 'info', 
                  icon: Optional[str] = None, collapsed: bool = False) -> Union[widgets.HTML, widgets.Accordion]:
    """
    Buat info box yang dapat di-collapse.
    
    Args:
        title: Judul info box
        content: Konten HTML info box
        style: Jenis style ('info', 'success', 'warning', 'error')
        icon: Emoji icon opsional
        collapsed: Apakah info box collapsed secara default
        
    Returns:
        Widget HTML atau Accordion berisi info box
    """
    style_config = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    icon_to_use = icon if icon else style_config['icon']
    title_with_icon = f"{icon_to_use} {title}"
    
    if collapsed:
        # Gunakan Accordion jika perlu collapsible
        content_widget = widgets.HTML(value=content)
        accordion = widgets.Accordion([content_widget])
        accordion.set_title(0, title_with_icon)
        accordion.selected_index = None
        return accordion
    else:
        # Gunakan HTML biasa jika tidak perlu collapsible
        box_html = f"""
        <div style="padding: 10px; background-color: {style_config['bg_color']}; 
                 border-left: 4px solid {style_config['border_color']}; 
                 color: {style_config['text_color']}; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top: 0; color: inherit;">{title_with_icon}</h4>
            {content}
        </div>
        """
        return widgets.HTML(value=box_html)

# Tab Creation
def create_tab_view(tabs: Dict[str, widgets.Widget]) -> widgets.Tab:
    """
    Buat komponen Tab dengan konfigurasi otomatis.
    
    Args:
        tabs: Dictionary berisi {nama_tab: widget_konten}
        
    Returns:
        Widget Tab yang dikonfigurasi
    """
    # Buat tab
    tab = widgets.Tab(children=list(tabs.values()))
    tab.layout = widgets.Layout(width='100%', margin='10px 0', overflow='visible')
    
    # Set judul tab
    for i, title in enumerate(tabs.keys()):
        tab.set_title(i, title)
    
    return tab

# Loading indicators
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

# Output area updates
def update_output_area(output_widget: widgets.Output, message: str, status: str = 'info', clear: bool = False):
    """
    Update area output dengan status baru.
    
    Args:
        output_widget: Widget output untuk diupdate
        message: Pesan untuk ditampilkan
        status: Jenis status ('info', 'success', 'warning', 'error')
        clear: Apakah perlu clear output sebelumnya
    """
    with output_widget:
        if clear:
            clear_output(wait=True)
        display(create_status_indicator(status, message))

# Observer registration
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

# File display
def display_file_info(file_path: str, description: Optional[str] = None) -> widgets.HTML:
    """
    Tampilkan informasi file dalam box informatif.
    
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
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size/1024:.1f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.1f} MB"
        
        # Format waktu
        time_str = datetime.datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Buat HTML
        html = f"""
        <div style="padding: 10px; background-color: {COLORS['light']}; border-radius: 5px; margin: 10px 0;">
            <p><strong>{ICONS['file']} File:</strong> {path.name}</p>
            <p><strong>{ICONS['folder']} Path:</strong> {path.parent}</p>
            <p><strong>📏 Size:</strong> {size_str}</p>
            <p><strong>{ICONS['time']} Modified:</strong> {time_str}</p>
        """
        
        if description:
            html += f"<p><strong>📝 Description:</strong> {description}</p>"
        
        html += "</div>"
        
        return widgets.HTML(value=html)
    else:
        return widgets.HTML(value=f"<p>{ICONS['warning']} File tidak ditemukan: {file_path}</p>")

# Progress updater
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
            # Update persentase
            progress_pct = int(value * 100 / total) if total > 0 else 0
            progress_bar.description = f"{progress_pct}%"
    
    return update_progress

# Task execution
def run_task(task_func, on_complete=None, on_error=None, with_output=None):
    """
    Jalankan task secara langsung dengan penanganan callback.
    
    Args:
        task_func: Fungsi task yang akan dijalankan
        on_complete: Callback saat task selesai
        on_error: Callback saat task error
        with_output: Widget output untuk menampilkan error
        
    Returns:
        Hasil dari task_func atau None jika terjadi error
    """
    try:
        # Run task langsung
        result = task_func()
        
        # Handle completion
        if on_complete:
            on_complete(result)
            
        return result
    except Exception as e:
        # Handle error
        if on_error:
            on_error(e)
        elif with_output:
            with with_output:
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        
        # Re-raise exception untuk penanganan lebih lanjut jika diperlukan
        raise

# Button groups
def create_button_group(buttons, layout=None):
    """
    Buat group tombol dengan layout konsisten.
    
    Args:
        buttons: List of tuples (label, style, icon, callback)
        layout: Layout opsional untuk container
        
    Returns:
        Widget HBox berisi tombol-tombol
    """
    btn_widgets = []
    
    for label, style, icon, callback in buttons:
        btn = widgets.Button(
            description=label,
            button_style=style,
            icon=icon,
            layout=widgets.Layout(margin='5px')
        )
        
        if callback:
            btn.on_click(callback)
            
        btn_widgets.append(btn)
    
    default_layout = widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        width='100%'
    )
    
    return widgets.HBox(
        btn_widgets, 
        layout=layout or default_layout
    )

# Confirmation dialog
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
    from smartcash.ui.utils.constants import FONTS
    
    # Dialog content
    content = widgets.VBox([
        widgets.HTML(f"""
        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
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
                layout=widgets.Layout(margin='5px'),
                tooltip="Batalkan operasi"
            ),
            widgets.Button(
                description=confirm_label,
                button_style="danger",
                icon='check',
                layout=widgets.Layout(margin='5px'),
                tooltip="Konfirmasi operasi"
            )
        ], layout=widgets.Layout(display='flex'))
    ], layout=widgets.Layout(padding='15px', border='1px solid #ddd', border_radius='4px'))
    
    # Set callbacks
    content.children[1].children[0].on_click(
        lambda b: on_cancel() if on_cancel else None
    )
    content.children[1].children[1].on_click(
        lambda b: on_confirm()
    )
    
    return content

# UI Elements
def create_divider() -> widgets.HTML:
    """Buat divider horizontal."""
    return widgets.HTML(f"<hr style='margin: 15px 0; border: 0; border-top: 1px solid {COLORS['border']};'>")

def create_spacing(height: str = '10px') -> widgets.HTML:
    """Buat elemen spacing untuk mengatur jarak antar komponen."""
    return widgets.HTML(f"<div style='height: {height};'></div>")

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
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f}{units[i]}"