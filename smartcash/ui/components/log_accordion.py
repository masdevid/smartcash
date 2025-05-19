"""
File: smartcash/ui/components/log_accordion.py
Deskripsi: Komponen log accordion yang dapat digunakan kembali
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def create_log_accordion(
    module_name: str = 'process',
    height: str = '200px',
    width: str = '100%',
    output_widget: Optional[widgets.Output] = None
) -> Dict[str, widgets.Widget]:
    """
    Membuat komponen log accordion yang dapat digunakan kembali.
    
    Args:
        module_name: Nama modul untuk label accordion
        height: Tinggi output log
        width: Lebar komponen
        output_widget: Widget output yang sudah ada (opsional)
        
    Returns:
        Dictionary berisi komponen log accordion
    """
    # Buat output widget jika tidak disediakan
    if output_widget is None:
        output_widget = widgets.Output(
            layout=widgets.Layout(
                max_height=height,
                overflow='auto',
                border='1px solid #ddd',
                padding='10px'
            )
        )
    
    # Buat accordion untuk log
    log_accordion = widgets.Accordion(
        children=[output_widget],
        layout=widgets.Layout(width=width, margin='10px 0')
    )
    
    # Set judul accordion
    log_accordion.set_title(0, f"{ICONS.get('log', '📋')} Log {module_name.capitalize()}")
    
    # Kembalikan dictionary berisi komponen
    return {
        'log_output': output_widget,
        'log_accordion': log_accordion
    }

def update_log(
    ui_components: Dict[str, Any],
    message: str,
    expand: bool = False,
    clear: bool = False
) -> None:
    """
    Update log dengan pesan baru.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan
        expand: Apakah perlu expand accordion
        clear: Apakah perlu clear output sebelumnya
    """
    # Cek apakah komponen log tersedia
    if 'log_output' not in ui_components:
        return
    
    # Update log
    with ui_components['log_output']:
        if clear:
            from IPython.display import clear_output
            clear_output(wait=True)
        display(widgets.HTML(f"<p>{message}</p>"))
    
    # Expand accordion jika diperlukan
    if expand and 'log_accordion' in ui_components:
        ui_components['log_accordion'].selected_index = 0
