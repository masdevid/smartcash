"""
File: smartcash/ui/training_config/hyperparameters/components/main_components.py
Deskripsi: Komponen utama yang mengintegrasikan semua komponen UI hyperparameter
"""

from typing import Dict, Any, List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.common.logger import get_logger
from smartcash.ui.components.tab_factory import create_tab_widget
from smartcash.ui.info_boxes.hyperparameters_info import (
    get_hyperparameters_info,
    get_basic_hyperparameters_info,
    get_optimization_hyperparameters_info,
    get_advanced_hyperparameters_info
)

from smartcash.ui.training_config.hyperparameters.components.basic_components import create_hyperparameters_basic_components
from smartcash.ui.training_config.hyperparameters.components.optimization_components import create_hyperparameters_optimization_components
from smartcash.ui.training_config.hyperparameters.components.advanced_components import create_hyperparameters_advanced_components
from smartcash.ui.training_config.hyperparameters.components.button_components import create_hyperparameters_button_components
from smartcash.ui.training_config.hyperparameters.components.info_panel_components import create_hyperparameters_info_panel

logger = get_logger(__name__)

def create_hyperparameters_ui_components() -> Dict[str, Any]:
    """
    Membuat semua komponen UI untuk hyperparameter.
    
    Returns:
        Dict berisi semua komponen UI
    """
    ui_components = {}
    
    # Buat komponen dasar
    basic_components = create_hyperparameters_basic_components()
    ui_components.update(basic_components)
    
    # Buat komponen optimasi
    optimization_components = create_hyperparameters_optimization_components()
    ui_components.update(optimization_components)
    
    # Buat komponen lanjutan
    advanced_components = create_hyperparameters_advanced_components()
    ui_components.update(advanced_components)
    
    # Buat komponen tombol
    button_components = create_hyperparameters_button_components()
    ui_components.update(button_components)
    
    # Buat panel informasi
    info_panel, update_info_func = create_hyperparameters_info_panel()
    ui_components['info_panel'] = info_panel
    ui_components['update_hyperparameters_info'] = update_info_func
    
    # Buat header
    header = create_header(
        title="Konfigurasi Hyperparameter",
        description="Pengaturan parameter untuk proses training model",
        icon=ICONS.get('settings', '⚙️')
    )
    
    # Buat button container untuk kompatibilitas dengan tes
    ui_components['button_container'] = widgets.HBox([
        button_components['save_button'],
        button_components['reset_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row nowrap',
        justify_content='flex-end',
        align_items='center',
        gap='10px',
        width='auto',
        margin='10px 0px'
    ))
    
    # Tambahkan sync_info ke ui_components untuk kompatibilitas dengan tes
    # Periksa apakah sync_info ada dalam button_components (untuk tes)
    ui_components['sync_info'] = button_components.get('sync_info', widgets.HTML(
        value=f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('info', 'ℹ️')} Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.</div>"
    ))
    
    # Buat form container untuk tab konfigurasi dengan 3 kolom sejajar
    form_container = widgets.VBox([
        widgets.HBox([
            widgets.Box([basic_components['basic_box']], layout=widgets.Layout(width='32%', height='100%', overflow='visible')),
            widgets.Box([optimization_components['optimization_box']], layout=widgets.Layout(width='32%', height='100%', overflow='visible')),
            widgets.Box([advanced_components['advanced_box']], layout=widgets.Layout(width='32%', height='100%', overflow='visible'))
        ], layout=widgets.Layout(
            width='auto',
            display='flex',
            flex_flow='row nowrap',
            align_items='flex-start',
            justify_content='space-between',
            overflow='visible'
        )),
        widgets.HBox([
            ui_components['button_container']
        ], layout=widgets.Layout(width='auto', justify_content='flex-end')),
        ui_components['sync_info']
    ], layout=widgets.Layout(width='auto', overflow='visible'))
    
    # Buat info box umum untuk hyperparameter
    general_info = get_hyperparameters_info(open_by_default=True)
    
    # Buat info container untuk tab informasi
    info_container = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('info', 'ℹ️')} Informasi Hyperparameter</h4>"),
        general_info,
        info_panel
    ], layout=widgets.Layout(width='auto', overflow='visible'))
    
    # Buat tab untuk form dan info
    tab_items = [
        ('Konfigurasi', form_container),
        ('Informasi', info_container)
    ]
    tabs = create_tab_widget(tab_items)
    
    # Set tab yang aktif
    tabs.selected_index = 0
    
    # Buat info boxes untuk footer
    basic_info = get_basic_hyperparameters_info(open_by_default=False)
    optimization_info = get_optimization_hyperparameters_info(open_by_default=False)
    advanced_info = get_advanced_hyperparameters_info(open_by_default=False)
    
    # Buat footer dengan info boxes yang menumpuk (stacked)
    # Set accordion behavior agar hanya satu yang terbuka
    basic_info.selected_index = None
    optimization_info.selected_index = None
    advanced_info.selected_index = None
    
    # Fungsi untuk menutup accordion lain saat satu dibuka
    def on_accordion_select(change, accordion_list):
        if change['new'] is not None:  # Jika ada yang dibuka
            # Tutup semua accordion lain
            for acc in accordion_list:
                if acc != change['owner']:
                    acc.selected_index = None
    
    # Daftar semua accordion
    accordion_list = [basic_info, optimization_info, advanced_info]
    
    # Tambahkan observer ke masing-masing accordion
    for acc in accordion_list:
        acc.observe(lambda change, acc_list=accordion_list: on_accordion_select(change, acc_list), names='selected_index')
    
    # Buat footer dengan info boxes yang menumpuk (stacked)
    footer_info = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('info', 'ℹ️')} Informasi Parameter</h4>"),
        widgets.VBox([
            basic_info,
            optimization_info,
            advanced_info
        ], layout=widgets.Layout(
            width='auto',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            overflow='visible'
        ))
    ], layout=widgets.Layout(
        width='auto',
        margin='20px 0 0 0',
        padding='10px',
        border_top='1px solid #ddd',
        overflow='visible'
    ))
    
    # Buat container utama
    main_container = widgets.VBox([
        header,
        tabs,
        button_components['status'],
        footer_info
    ], layout=widgets.Layout(width='auto', padding='10px', overflow='visible'))
    
    ui_components['main_container'] = main_container
    ui_components['main_layout'] = main_container  # Untuk kompatibilitas
    ui_components['tabs'] = tabs
    ui_components['header'] = header
    
    return ui_components
