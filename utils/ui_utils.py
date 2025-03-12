"""
File: smartcash/utils/ui_utils.py
Author: Alfrida Sabar
Deskripsi: Utilitas umum untuk komponen UI dan interaksi dengan user.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def create_header(title: str, description: Optional[str] = None) -> widgets.HTML:
    """
    Buat header dengan styling yang konsisten.
    
    Args:
        title: Judul header
        description: Deskripsi opsional
        
    Returns:
        Widget HTML yang berisi header
    """
    header = f'<h2 style="color: #3498db; margin-bottom: 10px;">{title}</h2>'
    if description:
        header += f'<p style="color: #555; margin-bottom: 15px;">{description}</p>'
    
    return widgets.HTML(value=header)

def create_section_title(title: str, icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat judul section dengan styling yang konsisten.
    
    Args:
        title: Judul section
        icon: Icon emoji opsional
        
    Returns:
        Widget HTML yang berisi judul section
    """
    title_with_icon = f"{icon} {title}" if icon else title
    return widgets.HTML(f'<h3 style="color: #2980b9; margin-top: 15px; margin-bottom: 10px;">{title_with_icon}</h3>')

def create_info_alert(message: str, 
                     alert_type: str = 'info',
                     icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat alert informasi dengan styling yang konsisten.
    
    Args:
        message: Pesan alert
        alert_type: Tipe alert ('info', 'success', 'warning', 'danger')
        icon: Icon emoji opsional
        
    Returns:
        Widget HTML yang berisi alert
    """
    # Tentukan warna dan ikon berdasarkan tipe alert
    alert_styles = {
        'info': {'color': '#0c5460', 'bg': '#d1ecf1', 'border': '#bee5eb', 'default_icon': 'ℹ️'},
        'success': {'color': '#155724', 'bg': '#d4edda', 'border': '#c3e6cb', 'default_icon': '✅'},
        'warning': {'color': '#856404', 'bg': '#fff3cd', 'border': '#ffeeba', 'default_icon': '⚠️'},
        'danger': {'color': '#721c24', 'bg': '#f8d7da', 'border': '#f5c6cb', 'default_icon': '❌'}
    }
    
    style = alert_styles.get(alert_type, alert_styles['info'])
    icon_to_use = icon if icon else style['default_icon']
    
    alert_html = f"""
    <div style="padding: 12px 15px; margin: 10px 0; 
               background-color: {style['bg']}; 
               color: {style['color']}; 
               border-left: 4px solid {style['border']};
               border-radius: 4px;">
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 10px; font-size: 1.2em;">{icon_to_use}</div>
            <div>{message}</div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=alert_html)

def create_status_indicator(status: str, message: str) -> widgets.HTML:
    """
    Buat indikator status dengan styling yang konsisten.
    
    Args:
        status: Tipe status ('success', 'warning', 'error', 'info')
        message: Pesan status
        
    Returns:
        Widget HTML yang berisi indikator status
    """
    status_styles = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    style = status_styles.get(status, status_styles['info'])
    
    status_html = f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: #f8f9fa;">
        <span style="color: {style['color']}; font-weight: bold;"> 
            {style['icon']} {message}
        </span>
    </div>
    """
    
    return widgets.HTML(value=status_html)

def styled_html(content: str, style: Optional[Dict[str, str]] = None) -> widgets.HTML:
    """
    Buat widget HTML dengan styling yang konsisten.
    
    Args:
        content: Konten HTML
        style: Optional dictionary berisi CSS styling
        
    Returns:
        Widget HTML yang sudah di-styling
    """
    if not style:
        return widgets.HTML(value=content)
    
    # Bentuk string CSS dari dictionary
    style_str = '; '.join([f"{k}: {v}" for k, v in style.items()])
    
    # Wrap konten dengan div yang memiliki styling
    styled_content = f'<div style="{style_str}">{content}</div>'
    
    return widgets.HTML(value=styled_content)

def create_alert(message: str, 
                alert_type: str = 'info',
                icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat alert box dengan styling yang konsisten.
    
    Args:
        message: Pesan yang akan ditampilkan
        alert_type: Tipe alert ('info', 'success', 'warning', 'danger')
        icon: Optional emoji icon
        
    Returns:
        Widget HTML berisi alert yang di-styling
    """
    # Set warna dan background berdasarkan tipe alert
    styles = {
        'info': {
            'bg_color': '#e8f4f8',
            'text_color': '#0c5460',
            'border_color': '#bee5eb',
            'default_icon': 'ℹ️'
        },
        'success': {
            'bg_color': '#d4edda',
            'text_color': '#155724',
            'border_color': '#c3e6cb',
            'default_icon': '✅'
        },
        'warning': {
            'bg_color': '#fff3cd',
            'text_color': '#856404',
            'border_color': '#ffeeba',
            'default_icon': '⚠️'
        },
        'danger': {
            'bg_color': '#f8d7da',
            'text_color': '#721c24',
            'border_color': '#f5c6cb',
            'default_icon': '❌'
        }
    }
    
    # Default ke info jika tipe tidak valid
    style = styles.get(alert_type, styles['info'])
    
    # Gunakan icon yang disediakan atau default
    icon_str = icon if icon else style['default_icon']
    
    alert_html = f"""
    <div style="padding: 10px; 
                background-color: {style['bg_color']}; 
                color: {style['text_color']}; 
                border-left: 4px solid {style['border_color']}; 
                border-radius: 5px; 
                margin: 10px 0;">
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 10px; font-size: 1.2em;">{icon_str}</div>
            <div>{message}</div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=alert_html)

def create_info_box(title: str, 
                   content: str, 
                   style: str = 'info',
                   icon: Optional[str] = None,
                   collapsed: bool = False) -> Union[widgets.Accordion, widgets.HTML]:
    """
    Buat box informasi dengan styling yang konsisten.
    
    Args:
        title: Judul box
        content: Konten dalam box (dapat berisi HTML)
        style: Tipe styling ('info', 'success', 'warning', 'error')
        icon: Optional emoji icon
        collapsed: True jika box dilipat secara default (hanya untuk tipe Accordion)
        
    Returns:
        Widget box informasi
    """
    style_configs = {
        'info': {
            'bg_color': '#d1ecf1',
            'border_color': '#0c5460',
            'text_color': '#0c5460',
            'default_icon': 'ℹ️'
        },
        'warning': {
            'bg_color': '#fff3cd',
            'border_color': '#856404',
            'text_color': '#856404',
            'default_icon': '⚠️'
        },
        'success': {
            'bg_color': '#d4edda',
            'border_color': '#155724',
            'text_color': '#155724',
            'default_icon': '✅'
        },
        'error': {
            'bg_color': '#f8d7da',
            'border_color': '#721c24',
            'text_color': '#721c24',
            'default_icon': '❌'
        }
    }
    
    style_config = style_configs.get(style, style_configs['info'])
    icon_to_use = icon if icon else style_config['default_icon']
    title_with_icon = f"{icon_to_use} {title}"
    
    if collapsed:
        # Gunakan Accordion jika perlu collapsible
        content_widget = widgets.HTML(value=content)
        accordion = widgets.Accordion([content_widget])
        accordion.set_title(0, title_with_icon)
        accordion.selected_index = None if collapsed else 0
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

def create_metric_display(label: str, 
                         value: Union[int, float, str],
                         unit: Optional[str] = None,
                         is_good: Optional[bool] = None) -> widgets.HTML:
    """
    Buat tampilan metrik dengan label dan nilai.
    
    Args:
        label: Label metrik
        value: Nilai metrik
        unit: Optional unit (e.g., '%', 'MB')
        is_good: True jika nilai baik, False jika buruk, None jika netral
        
    Returns:
        Widget HTML berisi metrik yang di-styling
    """
    # Tentukan warna berdasarkan nilai is_good
    if is_good is None:
        color = "#333333"  # Neutral
    elif is_good:
        color = "#28a745"  # Green for good
    else:
        color = "#dc3545"  # Red for bad
    
    # Format nilai
    if isinstance(value, float):
        formatted_value = f"{value:.2f}"
    else:
        formatted_value = str(value)
        
    # Tambahkan unit jika ada
    if unit:
        formatted_value = f"{formatted_value} {unit}"
    
    # Buat HTML
    metric_html = f"""
    <div style="margin: 10px 0; padding: 8px;">
        <div style="font-size: 0.9em; color: #666;">{label}</div>
        <div style="font-size: 1.5em; font-weight: bold; color: {color};">{formatted_value}</div>
    </div>
    """
    
    return widgets.HTML(value=metric_html)

def create_component_header(title: str, description: str = "", icon: str = "🔧") -> widgets.HTML:
    """
    Buat header komponen UI dengan styling konsisten.
    
    Args:
        title: Judul header
        description: Deskripsi header (opsional)
        icon: Emoji ikon untuk header
        
    Returns:
        Widget HTML header
    """
    header_html = f"""
    <div style="background-color: #f0f8ff; padding: 15px; color: black; 
              border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #3498db;">
        <h2 style="color: inherit; margin-top: 0;">{icon} {title}</h2>
        <p style="color: inherit; margin-bottom: 0;">{description}</p>
    </div>
    """
    
    return widgets.HTML(value=header_html)

def create_section_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat header untuk section UI.
    
    Args:
        title: Judul section
        description: Optional deskripsi
        icon: Optional emoji icon
        
    Returns:
        Widget HTML untuk header section
    """
    # Tambahkan ikon jika disediakan
    title_with_icon = f"{icon} {title}" if icon else title
    
    header_html = f'<h3 style="margin-bottom: 10px; color: #3498db;">{title_with_icon}</h3>'
    
    if description:
        header_html += f'<p style="color: #666; margin-bottom: 15px;">{description}</p>'
    
    return widgets.HTML(value=header_html)

def create_tab_view(tabs: Dict[str, widgets.Widget]) -> widgets.Tab:
    """
    Buat tampilan tab.
    
    Args:
        tabs: Dictionary dengan nama tab sebagai key dan widget sebagai value
        
    Returns:
        Widget Tab yang berisi konten
    """
    # Buat tab
    tab = widgets.Tab(children=list(tabs.values()))
    
    # Set judul tab
    for i, title in enumerate(tabs.keys()):
        tab.set_title(i, title)
    
    return tab

def create_loading_indicator(message: str = "Memproses...") -> Tuple[widgets.HBox, Callable]:
    """
    Buat indikator loading yang dapat ditampilkan dan disembunyikan.
    
    Args:
        message: Pesan yang ditampilkan
        
    Returns:
        Tuple berisi widget loading dan fungsi untuk menampilkan/menyembunyikan
    """
    # Buat spinner
    spinner = widgets.HTML(
        value='<i class="fa fa-spinner fa-spin" style="font-size: 20px; color: #3498db;"></i>'
    )
    
    # Buat pesan
    message_widget = widgets.HTML(value=f'<span style="margin-left: 10px;">{message}</span>')
    
    # Buat container
    loading = widgets.HBox([spinner, message_widget])
    loading.layout.display = 'none'  # Hide by default
    
    # Fungsi untuk toggle loading
    def toggle_loading(show: bool = True):
        loading.layout.display = 'flex' if show else 'none'
    
    return loading, toggle_loading

def update_output_area(output_widget: widgets.Output, message: str, status: str = 'info', clear: bool = False):
    """
    Update area output dengan pesan status.
    
    Args:
        output_widget: Widget output untuk diupdate
        message: Pesan yang akan ditampilkan
        status: Jenis status ('success', 'warning', 'error', 'info')
        clear: Flag untuk membersihkan output sebelum menampilkan
    """
    with output_widget:
        if clear:
            output_widget.clear_output()
        display(create_status_indicator(status, message))

def register_observer_callback(observer_manager, event_type, output_widget, 
                            group_name="ui_observer_group"):
    """
    Daftarkan callback untuk observer yang menampilkan pesan ke widget output.
    
    Args:
        observer_manager: Instance ObserverManager
        event_type: Tipe event yang akan dimonitor
        output_widget: Widget output untuk menampilkan notifikasi
        group_name: Nama grup observer
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

def plot_statistics(
    data: pd.DataFrame, 
    title: str, 
    kind: str = 'bar', 
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> None:
    """
    Plot statistik dari DataFrame.
    
    Args:
        data: Data yang akan diplot
        title: Judul plot
        kind: Tipe plot ('bar', 'line', etc.)
        figsize: Ukuran figure
        **kwargs: Parameter tambahan untuk plot
    """
    plt.figure(figsize=figsize)
    
    data.plot(kind=kind, **kwargs)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List nama class
        normalize: True untuk normalize matrix
        title: Judul plot
        cmap: Colormap
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
               cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def create_result_table(
    data: Dict[str, Any],
    title: str = 'Results',
    highlight_max: bool = True
) -> None:
    """
    Buat tabel hasil dengan highlighting.
    
    Args:
        data: Dictionary data yang akan ditampilkan
        title: Judul tabel
        highlight_max: True untuk highlight nilai maksimum
    """
    # Konversi ke DataFrame
    df = pd.DataFrame(data)
    
    # Display judul
    display(HTML(f"<h3>{title}</h3>"))
    
    # Display tabel dengan styling
    if highlight_max:
        display(df.style.highlight_max(axis=0, color='lightgreen'))
    else:
        display(df)

def create_progress_updater(progress_bar: widgets.IntProgress) -> Callable:
    """
    Buat fungsi untuk update progress bar.
    
    Args:
        progress_bar: Widget progress bar
        
    Returns:
        Fungsi untuk update progress
    """
    def update_progress(value: int, total: int, message: Optional[str] = None):
        # Update value
        progress_bar.max = total
        progress_bar.value = value
        
        # Update description jika ada message
        if message:
            progress_bar.description = message
    
    return update_progress

def display_file_info(file_path: str, description: Optional[str] = None) -> widgets.HTML:
    """
    Tampilkan informasi file dengan styling yang baik.
    
    Args:
        file_path: Path ke file
        description: Optional deskripsi
        
    Returns:
        Widget HTML dengan informasi file
    """
    import os
    from pathlib import Path
    
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
        from datetime import datetime
        time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Buat HTML
        html = f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <p><strong>📄 File:</strong> {path.name}</p>
            <p><strong>📁 Path:</strong> {path.parent}</p>
            <p><strong>📏 Size:</strong> {size_str}</p>
            <p><strong>🕒 Modified:</strong> {time_str}</p>
        """
        
        if description:
            html += f"<p><strong>📝 Description:</strong> {description}</p>"
        
        html += "</div>"
        
        return widgets.HTML(value=html)
    else:
        return widgets.HTML(value=f"<p>⚠️ File not found: {file_path}</p>")

def create_component_header(title, description="", icon="🔧"):
    """
    Buat header untuk komponen UI.
    
    Args:
        title: Judul komponen
        description: Deskripsi komponen
        icon: Emoji icon
        
    Returns:
        Widget HTML berisi header
    """
    return widgets.HTML(f"""
    <div style="background-color: #f0f8ff; padding: 15px; color: black; 
              border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #3498db;">
        <h2 style="color: inherit; margin-top: 0;">{icon} {title}</h2>
        <p style="color: inherit; margin-bottom: 0;">{description}</p>
    </div>
    """)

def create_info_box(title, content, style='info', icon=None, collapsed=False):
    """
    Buat box informasi dengan styling yang konsisten.
    
    Args:
        title: Judul box
        content: Konten box (html)
        style: Tipe styling ('info', 'success', 'warning', 'error')
        icon: Optional emoji icon
        collapsed: True jika box dilipat secara default
        
    Returns:
        Widget HTML untuk info box
    """
    style_configs = {
        'info': {
            'bg_color': '#d1ecf1',
            'border_color': '#0c5460',
            'text_color': '#0c5460',
            'default_icon': 'ℹ️'
        },
        'warning': {
            'bg_color': '#fff3cd',
            'border_color': '#856404',
            'text_color': '#856404',
            'default_icon': '⚠️'
        },
        'success': {
            'bg_color': '#d4edda',
            'border_color': '#155724',
            'text_color': '#155724',
            'default_icon': '✅'
        },
        'error': {
            'bg_color': '#f8d7da',
            'border_color': '#721c24',
            'text_color': '#721c24',
            'default_icon': '❌'
        }
    }
    
    style_config = style_configs.get(style, style_configs['info'])
    icon_to_use = icon if icon else style_config['default_icon']
    title_with_icon = f"{icon_to_use} {title}"
    
    if collapsed:
        # Gunakan Accordion jika perlu collapsible
        content_widget = widgets.HTML(value=content)
        accordion = widgets.Accordion([content_widget])
        accordion.set_title(0, title_with_icon)
        accordion.selected_index = None if collapsed else 0
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