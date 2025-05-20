"""
File: smartcash/ui/dataset/split/handlers/status_handlers.py
Deskripsi: Handler untuk status panel di split dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.common.logger import get_logger
from smartcash.ui.utils.alert_utils import update_status_panel as utils_update_status_panel
from smartcash.ui.utils.constants import ALERT_STYLES, COLORS, ICONS

logger = get_logger(__name__)

def update_status_panel(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Update panel status dengan pesan terbaru.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan (success, error, info, warning)
    """
    if 'status_panel' not in ui_components:
        ui_components = add_status_panel(ui_components)
    
    # Gunakan fungsi yang sudah ada di utils
    utils_update_status_panel(ui_components['status_panel'], message, status)

def create_status_panel() -> widgets.HTML:
    """
    Buat panel status untuk konfigurasi.
    
    Returns:
        widgets.HTML: Panel status
    """
    # Dapatkan style berdasarkan tipe status
    style_info = ALERT_STYLES.get('info', {})
    bg_color = style_info.get('bg_color', COLORS['alert_info_bg'])
    text_color = style_info.get('text_color', COLORS['alert_info_text'])
    icon = style_info.get('icon', ICONS['info'])
    
    return widgets.HTML(
        value=f"""
        <div style="padding:10px; background-color:{bg_color}; 
                 color:{text_color}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {text_color}">
            <p style="margin:5px 0">{icon} Status konfigurasi akan ditampilkan di sini</p>
        </div>""",
        layout=widgets.Layout(width='100%')
    )

def add_status_panel(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tambahkan panel status ke UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Buat panel status jika belum ada
    if 'status_panel' not in ui_components:
        status_panel = create_status_panel()
        ui_components['status_panel'] = status_panel
        
        # Tambahkan ke UI jika ada container
        if 'footer_container' in ui_components:
            ui_components['footer_container'].children = (*ui_components['footer_container'].children, status_panel)
        elif 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            # Cari container yang tepat untuk menambahkan status panel
            found = False
            for i, child in enumerate(ui_components['ui'].children):
                if hasattr(child, 'description') and child.description == 'Tombol':
                    # Tambahkan status panel setelah container tombol
                    container = widgets.VBox([status_panel])
                    ui_components['ui'].children = (*ui_components['ui'].children[:i+1], container, *ui_components['ui'].children[i+1:])
                    found = True
                    break
            
            # Jika tidak ada container tombol, tambahkan di akhir
            if not found:
                container = widgets.VBox([status_panel])
                ui_components['ui'].children = (*ui_components['ui'].children, container)
    
    return ui_components 