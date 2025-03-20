"""
File: smartcash/ui/helpers/ui_helpers.py
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
from smartcash.ui.utils.alert_utils import create_status_indicator

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