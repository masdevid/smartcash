"""
File: smartcash/ui/components/helpers.py
Deskripsi: Helper functions untuk komponen UI dengan styling yang lebih konsisten
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path

from smartcash.ui.utils.constants import COLORS, BUTTON_STYLES, ICONS
from smartcash.ui.components.widget_layouts import BUTTON_LAYOUTS, CONTAINER_LAYOUTS, COMPONENT_LAYOUTS, GROUP_LAYOUTS

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
    tab.layout = COMPONENT_LAYOUTS['tabs']
    
    # Set judul tab
    for i, title in enumerate(tabs.keys()):
        tab.set_title(i, title)
    
    return tab

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
            margin='10px 0'
        )
    )
    
    def toggle_loading(show: bool = True, new_message: Optional[str] = None):
        loading.layout.display = 'flex' if show else 'none'
        if new_message is not None:
            message_widget.value = f'<span style="margin-left: 10px;">{new_message}</span>'
    
    return loading, toggle_loading

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
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size/1024:.1f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.1f} MB"
        
        # Format waktu
        time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        
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
            from smartcash.ui.components.alerts import create_status_indicator
            with with_output:
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        
        # Re-raise exception untuk penanganan lebih lanjut jika diperlukan
        raise

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
        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                     color:{COLORS['alert_warning_text']}; 
                     border-left:4px solid {COLORS['alert_warning_text']}; 
                     border-radius:4px; margin:10px 0;">
            <h4 style="margin-top:0;">{ICONS['warning']} {title}</h4>
            <p>{message}</p>
        </div>
        """),
        widgets.HBox([
            widgets.Button(
                description=confirm_label,
                button_style='success',
                icon='check',
                layout=BUTTON_LAYOUTS['small']
            ),
            widgets.Button(
                description=cancel_label,
                button_style='danger',
                icon='times',
                layout=BUTTON_LAYOUTS['small']
            )
        ], layout=GROUP_LAYOUTS['horizontal'])
    ], layout=CONTAINER_LAYOUTS['card'])
    
    # Set callbacks
    content.children[1].children[0].on_click(
        lambda b: on_confirm()
    )
    content.children[1].children[1].on_click(
        lambda b: on_cancel() if on_cancel else None
    )
    
    return content