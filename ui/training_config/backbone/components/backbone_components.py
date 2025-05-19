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
from smartcash.ui.components.tab_factory import create_tab_widget
from smartcash.ui.components.model_info_panel import create_model_info_panel
from smartcash.ui.components.feature_checkbox_group import create_feature_checkbox_group
from smartcash.ui.components.config_form import create_config_form
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.split_config import create_split_config
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
    
    # Menggunakan shared component config_form untuk form konfigurasi
    fields = [
        {
            'type': 'dropdown',
            'name': 'model_type',
            'description': 'Tipe Model:',
            'options': [('EfficientNet Basic', 'efficient_basic'), ('YOLOv5s', 'yolov5s')],
            'value': 'efficient_basic'
        },
        {
            'type': 'dropdown',
            'name': 'backbone',
            'description': 'Backbone:',
            'options': [('EfficientNet-B4', 'efficientnet_b4'), ('CSPDarknet-S', 'cspdarknet_s')],
            'value': 'efficientnet_b4'
        }
    ]
    
    config_form_components = create_config_form(
        fields=fields,
        title="Konfigurasi Model",
        width="100%",
        icon="model",
        with_save_button=False,  # Gunakan save_reset_buttons yang terpisah
        with_reset_button=False  # Gunakan save_reset_buttons yang terpisah
    )
    
    # Menggunakan shared component feature_checkbox_group untuk fitur optimasi
    features = [
        ("Gunakan FeatureAdapter (Attention)", False),
        ("Gunakan ResidualAdapter (Residual)", False),
        ("Gunakan CIoU Loss", False)
    ]
    
    feature_group = create_feature_checkbox_group(
        features=features,
        title="Fitur Optimasi Model",
        width="100%",
        icon="settings"
    )
    
    # Buat tombol save dan reset menggunakan shared component
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi backbone dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi backbone ke default",
        save_icon="save",
        reset_icon="reset",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        button_width="100px"
    )
    
    # Buat panel untuk status
    status_panel = widgets.Output(
        layout=widgets.Layout(width='100%', min_height='50px')
    )
    
    # Menggunakan shared component model_info_panel untuk informasi backbone
    info_panel_components = create_model_info_panel(
        title="Informasi Backbone",
        min_height="150px",
        width="100%",
        icon="info"
    )
    
    # Buat container untuk form
    form_container = widgets.VBox([
        config_form_components['container'],
        widgets.HTML("<hr style='margin: 10px 0px; border-style: dashed;'>"),
        feature_group['container'],
        widgets.HTML("<hr style='margin: 10px 0px; border-style: dashed;'>"),
        save_reset_buttons['container']
    ])
    
    # Buat container untuk info
    info_container = widgets.VBox([
        info_panel_components['container']
    ])
    
    # Buat tab untuk form dan info
    tab_items = [
        ('Konfigurasi', form_container),
        ('Informasi', info_container)
    ]
    tabs = create_tab_widget(tab_items)
    
    # Set tab yang aktif
    tabs.selected_index = 0
    
    # Buat container utama
    main_container = widgets.VBox([
        header,
        tabs,
        status_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Kumpulkan semua komponen UI
    ui_components = {
        'main_container': main_container,
        'model_type_dropdown': config_form_components['fields']['model_type'],
        'backbone_dropdown': config_form_components['fields']['backbone'],
        'use_attention_checkbox': feature_group['checkboxes']['gunakan_featureadapter_attention'],
        'use_residual_checkbox': feature_group['checkboxes']['gunakan_residualadapter_residual'],
        'use_ciou_checkbox': feature_group['checkboxes']['gunakan_ciou_loss'],
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'button_container': save_reset_buttons['button_container'],
        'sync_info': save_reset_buttons['sync_info'],
        'status_panel': status_panel,
        'info_panel': info_panel_components['info_panel'],
        'tabs': tabs,
        # Tambahkan referensi ke komponen shared untuk akses lebih mudah
        'config_form': config_form_components,
        'feature_group': feature_group,
        'info_panel_components': info_panel_components,
        'save_reset_buttons': save_reset_buttons
    }
    
    return ui_components
