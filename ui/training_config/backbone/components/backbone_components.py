"""
File: smartcash/ui/training_config/backbone/components/backbone_components.py
Deskripsi: Komponen UI untuk pemilihan backbone model SmartCash
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.tab_factory import create_tabs
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def create_backbone_ui() -> Dict[str, Any]:
    """
    Membuat komponen UI untuk pemilihan backbone model.
    
    Returns:
        Dictionary berisi komponen UI yang telah diinisialisasi
    """
    # Buat header
    header = create_header(
        title="Konfigurasi Backbone Model",
        description="Pilih backbone dan konfigurasi model untuk deteksi mata uang",
        icon=ICONS.get('model', '🧠')
    )
    
    # Buat dropdown untuk tipe model
    model_type_dropdown = widgets.Dropdown(
        options=[
            ('EfficientNet Basic', 'efficient_basic'),
            ('YOLOv5s', 'yolov5s')
        ],
        value='efficient_basic',
        description='Tipe Model:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat dropdown untuk backbone
    backbone_dropdown = widgets.Dropdown(
        options=[
            ('EfficientNet-B4', 'efficientnet_b4'),
            ('CSPDarknet-S', 'cspdarknet_s')
        ],
        value='efficientnet_b4',
        description='Backbone:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat checkbox untuk fitur optimasi
    use_attention_checkbox = widgets.Checkbox(
        value=False,
        description='Gunakan FeatureAdapter (Attention)',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    use_residual_checkbox = widgets.Checkbox(
        value=False,
        description='Gunakan ResidualAdapter (Residual)',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    use_ciou_checkbox = widgets.Checkbox(
        value=False,
        description='Gunakan CIoU Loss',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat tombol Save dan Reset
    save_button = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='primary',
        icon=ICONS.get('save', '💾'),
        layout=widgets.Layout(width='auto')
    )
    
    reset_button = widgets.Button(
        description='Reset ke Default',
        button_style='warning',
        icon=ICONS.get('reset', '🔄'),
        layout=widgets.Layout(width='auto')
    )
    
    # Tambahkan keterangan sinkronisasi otomatis
    sync_info = widgets.HTML(
        value=f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('info', 'ℹ️')} Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.</div>"
    )
    
    # Buat panel untuk status
    status_panel = widgets.Output(
        layout=widgets.Layout(width='100%', min_height='50px')
    )
    
    # Buat panel untuk informasi backbone
    info_panel = widgets.Output(
        layout=widgets.Layout(width='100%', min_height='100px')
    )
    
    # Buat container untuk form
    form_container = widgets.VBox([
        widgets.HBox([model_type_dropdown], layout=widgets.Layout(width='100%', margin='10px 0px')),
        widgets.HBox([backbone_dropdown], layout=widgets.Layout(width='100%', margin='10px 0px')),
        widgets.HTML("<hr style='margin: 10px 0px; border-style: dashed;'>"),
        widgets.HTML("<h4>Fitur Optimasi Model</h4>"),
        widgets.VBox([
            use_attention_checkbox,
            use_residual_checkbox,
            use_ciou_checkbox
        ], layout=widgets.Layout(margin='10px 0px')),
        widgets.HTML("<hr style='margin: 10px 0px; border-style: dashed;'>"),
        widgets.HBox([save_button, reset_button], layout=widgets.Layout(justify_content='space-between', margin='20px 0px 10px 0px')),
        sync_info
    ])
    
    # Buat container untuk info
    info_container = widgets.VBox([
        widgets.HTML("<h4>Informasi Backbone</h4>"),
        info_panel
    ])
    
    # Buat tab untuk form dan info
    tabs = create_tabs(
        [form_container, info_container],
        ['Konfigurasi', 'Informasi'],
        selected_index=0
    )
    
    # Buat container utama
    main_container = widgets.VBox([
        header,
        tabs,
        status_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Kumpulkan semua komponen UI
    ui_components = {
        'main_container': main_container,
        'model_type_dropdown': model_type_dropdown,
        'backbone_dropdown': backbone_dropdown,
        'use_attention_checkbox': use_attention_checkbox,
        'use_residual_checkbox': use_residual_checkbox,
        'use_ciou_checkbox': use_ciou_checkbox,
        'save_button': save_button,
        'reset_button': reset_button,
        'sync_info': sync_info,
        'status_panel': status_panel,
        'info_panel': info_panel,
        'tabs': tabs
    }
    
    return ui_components
