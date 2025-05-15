"""
File: smartcash/ui/training_config/training_strategy/components/main_components.py
Deskripsi: Komponen utama yang mengintegrasikan semua komponen UI strategi pelatihan
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.utils.header_utils import create_header
from smartcash.common.logger import get_logger

from smartcash.ui.training_config.training_strategy.components.utils_components import create_training_strategy_utils_components
from smartcash.ui.training_config.training_strategy.components.validation_components import create_training_strategy_validation_components
from smartcash.ui.training_config.training_strategy.components.multiscale_components import create_training_strategy_multiscale_components
from smartcash.ui.training_config.training_strategy.components.button_components import create_training_strategy_button_components
from smartcash.ui.training_config.training_strategy.components.info_panel_components import create_training_strategy_info_panel

logger = get_logger(__name__)

def create_training_strategy_ui_components() -> Dict[str, Any]:
    """
    Membuat semua komponen UI untuk strategi pelatihan.
    
    Returns:
        Dict berisi semua komponen UI
    """
    ui_components = {}
    
    # Buat komponen utilitas
    utils_components = create_training_strategy_utils_components()
    ui_components.update(utils_components)
    
    # Buat komponen validasi
    validation_components = create_training_strategy_validation_components()
    ui_components.update(validation_components)
    
    # Buat komponen multi-scale
    multiscale_components = create_training_strategy_multiscale_components()
    ui_components.update(multiscale_components)
    
    # Buat komponen tombol
    button_components = create_training_strategy_button_components()
    ui_components.update(button_components)
    
    # Buat panel informasi
    info_panel, update_info_func = create_training_strategy_info_panel()
    ui_components['training_strategy_info'] = info_panel
    ui_components['update_training_strategy_info'] = update_info_func
    
    # Buat form container untuk tab konfigurasi
    ui_components['tabs'] = widgets.Tab()
    ui_components['tabs'].children = [
        utils_components['utils_box'], 
        validation_components['validation_box'], 
        multiscale_components['multiscale_box']
    ]
    
    # Set judul tab
    ui_components['tabs']._titles = {
        0: f"{ICONS.get('settings', '⚙️')} Utilitas Training",
        1: f"{ICONS.get('check', '✓')} Validasi",
        2: f"{ICONS.get('scale', '📏')} Multi-scale"
    }
    
    form_container = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('settings', '⚙️')} Konfigurasi Strategi Pelatihan</h4>"),
        ui_components['tabs'],
        widgets.HBox([
            button_components['button_container']
        ], layout=widgets.Layout(width='auto', justify_content='flex-end')),
        button_components['sync_info']
    ], layout=widgets.Layout(width='auto', overflow='visible'))
    
    # Buat container untuk info
    info_container = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('info', 'ℹ️')} Informasi Strategi Pelatihan</h4>"),
        info_panel
    ], layout=widgets.Layout(width='auto', overflow='visible'))
    
    # Buat tab untuk form dan info
    tab_items = [
        ('Konfigurasi', form_container),
        ('Informasi', info_container)
    ]
    ui_components['main_tabs'] = widgets.Tab()
    ui_components['main_tabs'].children = [item[1] for item in tab_items]
    for i, (title, _) in enumerate(tab_items):
        ui_components['main_tabs'].set_title(i, title)
    
    # Set tab yang aktif
    ui_components['tabs'].selected_index = 0
    
    # Buat header dengan komponen standar
    header = create_header(
        title="Konfigurasi Strategi Pelatihan",
        description="Pengaturan strategi pelatihan untuk model deteksi mata uang",
        icon=ICONS.get('training', '🏋️')
    )
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        header,
        ui_components['main_tabs'],
        button_components['status']
    ], layout=widgets.Layout(width='auto', padding='10px', overflow='visible'))
    
    # Tambahkan referensi komponen tambahan ke ui_components
    ui_components.update({
        'header': header,
        'module_name': 'training_strategy'
    })
    
    return ui_components
