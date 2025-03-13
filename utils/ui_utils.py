"""
File: smartcash/utils/ui_utils.py
Author: Refactored
Deskripsi: Utilitas terpadu untuk komponen UI dengan pendekatan DRY untuk reusable styling dan komponen.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import threading
from datetime import datetime

# ===== STYLE CONSTANTS =====

# Color palette
COLORS = {
    'primary': '#3498db',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'muted': '#6c757d',
    'highlight': '#e65100',
    
    # Alert colors
    'alert_info_bg': '#d1ecf1',
    'alert_info_text': '#0c5460',
    'alert_success_bg': '#d4edda',
    'alert_success_text': '#155724',
    'alert_warning_bg': '#fff3cd',
    'alert_warning_text': '#856404',
    'alert_danger_bg': '#f8d7da',
    'alert_danger_text': '#721c24',
}

# Emoji icons
ICONS = {
    'success': '✅',
    'warning': '⚠️',
    'error': '❌',
    'info': 'ℹ️',
    'config': '⚙️',
    'data': '📊',
    'processing': '🔄',
    'start': '🚀',
    'download': '📥',
    'upload': '📤',
    'save': '💾',
    'folder': '📁',
    'file': '📄',
    'model': '🧠',
    'time': '⏱️',
    'metric': '📈',
    'settings': '🔧',
    'tools': '🛠️',
    'split': '✂️',
    'augmentation': '🎨',
    'training': '🏋️',
    'evaluation': '🔍',
    'cleanup': '🧹',
    'stop': '🛑',
    'pause': '⏸️',
    'play': '▶️',
    'medal': '🏆',
}

# Alert styles
ALERT_STYLES = {
    'info': {
        'bg_color': COLORS['alert_info_bg'],
        'text_color': COLORS['alert_info_text'],
        'border_color': COLORS['alert_info_text'],
        'icon': ICONS['info']
    },
    'success': {
        'bg_color': COLORS['alert_success_bg'],
        'text_color': COLORS['alert_success_text'],
        'border_color': COLORS['alert_success_text'],
        'icon': ICONS['success']
    },
    'warning': {
        'bg_color': COLORS['alert_warning_bg'],
        'text_color': COLORS['alert_warning_text'],
        'border_color': COLORS['alert_warning_text'],
        'icon': ICONS['warning']
    },
    'error': {
        'bg_color': COLORS['alert_danger_bg'],
        'text_color': COLORS['alert_danger_text'],
        'border_color': COLORS['alert_danger_text'],
        'icon': ICONS['error']
    }
}

# Standard layouts
LAYOUTS = {
    'header': widgets.Layout(margin='0 0 15px 0'),
    'section': widgets.Layout(margin='15px 0 10px 0'),
    'container': widgets.Layout(width='100%', padding='10px'),
    'output': widgets.Layout(
        width='100%',
        border='1px solid #ddd',
        min_height='100px',
        max_height='300px',
        margin='10px 0',
        overflow='auto'
    ),
    'button': widgets.Layout(margin='10px 0'),
    'button_small': widgets.Layout(margin='5px'),
    'hbox': widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        width='100%'
    ),
    'vbox': widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%'
    )
}

# ===== CORE UI COMPONENTS =====

def create_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    # Tambahkan ikon jika disediakan
    title_with_icon = f"{icon} {title}" if icon else title
    
    header_html = f"""
    <div style="background-color: #f0f8ff; padding: 15px; color: black; 
            border-radius: 5px; margin-bottom: 15px; border-left: 5px solid {COLORS['primary']};">
        <h2 style="color: {COLORS['dark']}; margin-top: 0;">{title_with_icon}</h2>
    """
    
    if description:
        header_html += f'<p style="color: {COLORS["dark"]}; margin-bottom: 0;">{description}</p>'
    
    header_html += "</div>"
    
    return widgets.HTML(value=header_html)

def create_component_header(title, description="", icon="🔧"):
    """Alias untuk create_header dengan parameter yang dibalik untuk backward compatibility."""
    return create_header(title, description, icon)

def create_section_title(title: str, icon: Optional[str] = "") -> widgets.HTML:
    return widgets.HTML(f"""
    <h3 style="color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;">
        {icon} {title}
    </h3>
    """)

def create_info_box(title: str, content: str, style: str = 'info', 
                  icon: Optional[str] = None, collapsed: bool = False) -> Union[widgets.HTML, widgets.Accordion]:
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

def create_status_indicator(status: str, message: str) -> HTML:
    style_config = ALERT_STYLES.get(status, ALERT_STYLES['info'])
    
    return HTML(f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: {COLORS['light']};">
        <span style="color: {style_config['text_color']}; font-weight: bold;"> 
            {style_config['icon']} {message}
        </span>
    </div>
    """)

def create_alert(message: str, alert_type: str = 'info', icon: Optional[str] = None) -> widgets.HTML:
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

def create_info_alert(message, alert_type='info', icon=None):
    """Alias untuk create_alert untuk backward compatibility."""
    return create_alert(message, alert_type, icon)

def create_metric_display(label: str, 
                         value: Union[int, float, str],
                         unit: Optional[str] = None,
                         is_good: Optional[bool] = None) -> widgets.HTML:
    # Tentukan warna berdasarkan nilai is_good
    if is_good is None:
        color = COLORS['dark']  # Neutral
    elif is_good:
        color = COLORS['success']  # Green for good
    else:
        color = COLORS['danger']  # Red for bad
    
    # Format nilai
    if isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)
        
    # Tambahkan unit jika ada
    if unit:
        formatted_value = f"{formatted_value} {unit}"
    
    # Buat HTML
    metric_html = f"""
    <div style="margin: 10px 5px; padding: 8px; background-color: {COLORS['light']}; 
                border-radius: 5px; text-align: center; min-width: 120px;">
        <div style="font-size: 0.9em; color: {COLORS['muted']};">{label}</div>
        <div style="font-size: 1.3em; font-weight: bold; color: {color};">{formatted_value}</div>
    </div>
    """
    
    return widgets.HTML(value=metric_html)

def styled_html(content: str, bg_color: str = "#f8f9fa", text_color: str = "#2c3e50", 
              border_color: Optional[str] = None, padding: int = 10, margin: int = 10) -> widgets.HTML:
    border_style = f"border-left: 4px solid {border_color}; " if border_color else ""
    
    return widgets.HTML(f"""
    <div style="background-color: {bg_color}; color: {text_color}; 
                {border_style}padding: {padding}px; margin: {margin}px 0; 
                border-radius: 4px;">
        {content}
    </div>
    """)

# ===== UI HELPERS =====

def create_tab_view(tabs: Dict[str, widgets.Widget]) -> widgets.Tab:
    # Buat tab
    tab = widgets.Tab(children=list(tabs.values()))
    
    # Set judul tab
    for i, title in enumerate(tabs.keys()):
        tab.set_title(i, title)
    
    return tab

def create_loading_indicator(message: str = "Memproses...") -> Tuple[widgets.HBox, Callable]:
    spinner = widgets.HTML(
        value=f'<i class="fa fa-spinner fa-spin" style="font-size: 20px; color: {COLORS["primary"]};"></i>'
    )
    
    message_widget = widgets.HTML(value=f'<span style="margin-left: 10px;">{message}</span>')
    
    loading = widgets.HBox([spinner, message_widget])
    loading.layout.display = 'none'  # Hide by default
    
    def toggle_loading(show: bool = True):
        loading.layout.display = 'flex' if show else 'none'
    
    return loading, toggle_loading

def update_output_area(output_widget: widgets.Output, message: str, status: str = 'info', clear: bool = False):
    with output_widget:
        if clear:
            clear_output()
        display(create_status_indicator(status, message))

def register_observer_callback(observer_manager, event_type, output_widget, 
                            group_name="ui_observer_group"):
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

def display_file_info(file_path: str, description: Optional[str] = None) -> widgets.HTML:
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
        return widgets.HTML(value=f"<p>{ICONS['warning']} File not found: {file_path}</p>")

# ===== DATA VISUALIZATION =====

def plot_statistics(
    data: pd.DataFrame, 
    title: str, 
    kind: str = 'bar', 
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> None:
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
    import seaborn as sns
    
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
    def update_progress(value: int, total: int, message: Optional[str] = None):
        # Update value
        progress_bar.max = total
        progress_bar.value = value
        
        # Update description jika ada message
        if message:
            progress_bar.description = message
    
    return update_progress

# ===== THREADING HELPERS =====

def run_async_task(task_func, on_complete=None, on_error=None, with_output=None):
    def thread_task():
        result = None
        
        try:
            # Run task
            result = task_func()
            
            # Handle completion
            if on_complete:
                on_complete(result)
                
        except Exception as e:
            # Handle error
            if on_error:
                on_error(e)
            elif with_output:
                with with_output:
                    display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
    
    # Start thread
    thread = threading.Thread(target=thread_task)
    thread.daemon = True
    thread.start()
    
    return thread

# ===== WIDGET CREATION =====

def create_button_group(buttons, layout=None):
    style_map = {
        'primary': 'primary',
        'success': 'success',
        'info': 'info',
        'warning': 'warning',
        'danger': 'danger',
        'default': ''
    }
    
    btn_widgets = []
    
    for label, style, icon, callback in buttons:
        btn = widgets.Button(
            description=label,
            button_style=style_map.get(style, ''),
            icon=icon,
            layout=LAYOUTS['button_small']
        )
        
        if callback:
            btn.on_click(callback)
            
        btn_widgets.append(btn)
    
    return widgets.HBox(btn_widgets, layout=layout or LAYOUTS['hbox'])

def create_confirmation_dialog(title, message, on_confirm, on_cancel=None, 
                              confirm_label="Confirm", cancel_label="Cancel"):
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
                description=cancel_label,
                button_style="warning",
                layout=widgets.Layout(margin='0 10px 0 0')
            ),
            widgets.Button(
                description=confirm_label,
                button_style="danger",
                layout=widgets.Layout(margin='0')
            )
        ])
    ])
    
    # Set callbacks
    content.children[1].children[0].on_click(
        lambda b: on_cancel() if on_cancel else None
    )
    content.children[1].children[1].on_click(
        lambda b: on_confirm()
    )
    
    return content

# ===== COMPATIBILITY FUNCTIONS =====
# These functions provide backward compatibility with older code

# Backward compatibility
create_section_header = create_section_title