"""
File: smartcash/ui/dataset/preprocessing/utils/ui_helpers.py
Deskripsi: Fungsi-fungsi helper umum untuk UI preprocessing yang tidak terkait langsung dengan notifikasi, state, atau progress
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS, COLORS, ALERT_STYLES

def ensure_output_area(ui_components: Dict[str, Any], area_name: str, parent_name: Optional[str] = 'main_output') -> Dict[str, Any]:
    """
    Pastikan area output tertentu tersedia di ui_components.
    
    Args:
        ui_components: Dictionary komponen UI
        area_name: Nama area yang ingin dipastikan tersedia
        parent_name: Nama parent container (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Skip jika area sudah tersedia
    if area_name in ui_components and hasattr(ui_components[area_name], 'clear_output'):
        return ui_components
    
    # Buat area baru
    ui_components[area_name] = widgets.Output()
    
    # Tambahkan ke parent jika tersedia
    if parent_name in ui_components and hasattr(ui_components[parent_name], 'children'):
        # Tambahkan widget ke children dari parent
        parent = ui_components[parent_name]
        if hasattr(parent, 'children'):
            parent.children = list(parent.children) + [ui_components[area_name]]
    
    return ui_components

def create_message_dialog(
    message: str, 
    title: str, 
    message_type: str = 'info', 
    buttons: Optional[List[Dict[str, Any]]] = None
) -> widgets.VBox:
    """
    Buat dialog pesan dengan tombol dan styling yang konsisten.
    
    Args:
        message: Pesan yang akan ditampilkan
        title: Judul dialog
        message_type: Tipe pesan ('info', 'success', 'warning', 'error')
        buttons: List dictionary konfigurasi tombol (opsional)
        
    Returns:
        Widget dialog pesan
    """
    # Default buttons jika tidak disediakan
    if buttons is None:
        buttons = [
            {'label': 'OK', 'style': 'primary', 'callback': None}
        ]
    
    # Ambil style berdasarkan tipe
    style = ALERT_STYLES.get(message_type, ALERT_STYLES['info'])
    
    # Buat dialog content
    message_html = widgets.HTML(
        value=f"""
        <div style="padding:10px; background-color:{style['bg_color']}; 
                    color:{style['text_color']}; 
                    border-left:4px solid {style['border_color']}; 
                    border-radius:4px; margin:10px 0;">
            <h4 style="margin-top:0; color: inherit;">{style['icon']} {title}</h4>
            <p style="margin-bottom:0;">{message}</p>
        </div>
        """
    )
    
    # Buat tombol
    button_widgets = []
    for btn in buttons:
        button = widgets.Button(
            description=btn.get('label', 'OK'),
            button_style=btn.get('style', 'primary'),
            icon=btn.get('icon', ''),
            layout=widgets.Layout(margin='5px')
        )
        
        # Setup callback jika ada
        if 'callback' in btn and callable(btn['callback']):
            button.on_click(btn['callback'])
            
        button_widgets.append(button)
    
    # Buat container tombol
    button_box = widgets.HBox(
        button_widgets,
        layout=widgets.Layout(display='flex', justify_content='flex-end')
    )
    
    # Gabungkan ke dalam dialog
    dialog = widgets.VBox(
        [message_html, button_box],
        layout=widgets.Layout(padding='15px', border='1px solid #ddd', border_radius='4px')
    )
    
    return dialog

def toggle_widgets(widgets_list: List[widgets.Widget], disable: bool = True) -> None:
    """
    Toggle status disabled dari list widget.
    
    Args:
        widgets_list: List widget yang akan di-toggle
        disable: True untuk disable widget, False untuk enable
    """
    for widget in widgets_list:
        if hasattr(widget, 'disabled'):
            widget.disabled = disable

def get_widget_value(ui_components: Dict[str, Any], widget_name: str, default_value: Any = None) -> Any:
    """
    Dapatkan nilai dari widget dengan penanganan error yang baik.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_name: Nama widget yang ingin diambil nilainya
        default_value: Nilai default jika widget tidak ditemukan atau tidak memiliki value
        
    Returns:
        Nilai widget atau default value
    """
    if widget_name not in ui_components:
        return default_value
    
    widget = ui_components[widget_name]
    if not hasattr(widget, 'value'):
        return default_value
    
    return widget.value

def set_widget_value(ui_components: Dict[str, Any], widget_name: str, value: Any) -> bool:
    """
    Set nilai ke widget dengan penanganan error yang baik.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_name: Nama widget yang ingin diset nilainya
        value: Nilai yang akan diset
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    if widget_name not in ui_components:
        return False
    
    widget = ui_components[widget_name]
    if not hasattr(widget, 'value'):
        return False
    
    try:
        widget.value = value
        return True
    except Exception:
        return False

def collect_widget_values(ui_components: Dict[str, Any], widget_names: List[str]) -> Dict[str, Any]:
    """
    Kumpulkan nilai dari beberapa widget.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_names: List nama widget yang ingin dikumpulkan nilainya
        
    Returns:
        Dictionary dengan nama widget sebagai key dan nilainya sebagai value
    """
    result = {}
    for name in widget_names:
        result[name] = get_widget_value(ui_components, name)
    return result

def add_callback_to_widgets(ui_components: Dict[str, Any], widget_names: List[str], callback: Callable, event: str = 'on_click') -> None:
    """
    Tambahkan callback ke multiple widgets.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_names: List nama widget yang ingin ditambahkan callback
        callback: Fungsi callback
        event: Nama event (default: 'on_click')
    """
    for name in widget_names:
        if name in ui_components:
            widget = ui_components[name]
            if hasattr(widget, event) and callable(getattr(widget, event)):
                getattr(widget, event)(callback) 