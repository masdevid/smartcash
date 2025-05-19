"""
File: smartcash/ui/training_config/hyperparameters/components/button_components.py
Deskripsi: Komponen tombol untuk konfigurasi hyperparameter
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.sync_info_message import create_sync_info_message

def create_hyperparameters_button_components() -> Dict[str, Any]:
    """
    Membuat komponen tombol untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Gunakan shared component save_reset_buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi hyperparameter dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi hyperparameter ke default",
        save_icon="save",
        reset_icon="reset",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        button_width="100px"
    )
    
    # Tambahkan komponen ke ui_components
    ui_components['save_button'] = save_reset_buttons['save_button']
    ui_components['reset_button'] = save_reset_buttons['reset_button']
    ui_components['button_container'] = save_reset_buttons['button_container']
    ui_components['sync_info'] = save_reset_buttons['sync_info']
    ui_components['save_reset_buttons'] = save_reset_buttons  # Tambahkan referensi lengkap
    
    # Buat panel untuk status (seperti backbone)
    ui_components['status_panel'] = widgets.Output(
        layout=widgets.Layout(width='100%', min_height='50px')
    )
    
    # Untuk kompatibilitas dengan kode lama
    ui_components['status'] = ui_components['status_panel']
    
    return ui_components
