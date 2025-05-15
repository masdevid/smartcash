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
    
    # Buat tabs
    ui_components['tabs'] = widgets.Tab()
    ui_components['tabs'].children = [
        utils_components['utils_box'], 
        validation_components['validation_box'], 
        multiscale_components['multiscale_box']
    ]
    ui_components['tabs'].set_title(0, f"{ICONS.get('settings', '⚙️')} Utilitas Training")
    ui_components['tabs'].set_title(1, f"{ICONS.get('check', '✓')} Validasi")
    ui_components['tabs'].set_title(2, f"{ICONS.get('scale', '📏')} Multi-scale")
    
    # Buat info box
    ui_components['info_box'] = widgets.VBox(
        [info_panel],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    # Header dengan komponen standar
    header = create_header(
        title=f"{ICONS.get('training', '🏋️')} Training Strategy Configuration",
        description="Konfigurasi strategi pelatihan untuk model deteksi mata uang"
    )
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS.get('info', 'ℹ️')} Konfigurasi strategi pelatihan model</p>
        </div>"""
    )
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[button_components['status']], selected_index=None)
    log_accordion.set_title(0, f"{ICONS.get('file', '📄')} Training Strategy Logs")
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('settings', '⚙️')} Training Strategy</h4>"),
        ui_components['tabs'],
        ui_components['info_box'],
        create_divider(),
        button_components['button_container'],
        log_accordion
    ])
    
    # Tambahkan referensi komponen tambahan ke ui_components
    ui_components.update({
        'header': header,
        'status_panel': status_panel,
        'log_accordion': log_accordion,
        'module_name': 'training_strategy'
    })
    
    return ui_components
