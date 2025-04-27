"""
File: smartcash/ui/training_config/backbone/components/backbone_components.py
Deskripsi: Komponen UI untuk pemilihan backbone model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def create_backbone_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk pemilihan backbone model.
    
    Args:
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI
    """
    # Inisialisasi komponen
    ui_components = {}
    
    # Dapatkan konfigurasi backbone yang tersedia
    from smartcash.model.config.backbone_config import BackboneConfig
    
    # Daftar backbone yang didukung
    backbone_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value="<h3>🔄 Konfigurasi Backbone Model</h3>"
    )
    
    ui_components['backbone_type'] = widgets.Dropdown(
        options=backbone_options,
        value='efficientnet_b4',
        description='Backbone:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['pretrained'] = widgets.Checkbox(
        value=True,
        description='Gunakan pretrained weights',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['freeze_backbone'] = widgets.Checkbox(
        value=True,
        description='Freeze backbone layers',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['freeze_layers'] = widgets.IntSlider(
        value=3,
        min=0,
        max=5,
        step=1,
        description='Freeze layers:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Informasi backbone
    ui_components['backbone_info'] = widgets.HTML(
        value="<p>Informasi backbone akan ditampilkan di sini</p>"
    )
    
    # Buat tombol aksi
    ui_components['save_button'] = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='success',
        icon='save',
        tooltip='Simpan konfigurasi backbone',
        layout=widgets.Layout(width='200px')
    )
    
    ui_components['reset_button'] = widgets.Button(
        description='Reset',
        button_style='warning',
        icon='refresh',
        tooltip='Reset ke default',
        layout=widgets.Layout(width='100px')
    )
    
    # Status indicator
    ui_components['status'] = widgets.Output(
        layout=widgets.Layout(width='100%', padding='10px')
    )
    
    # Susun layout
    ui_components['form_items'] = [
        ui_components['backbone_type'],
        ui_components['pretrained'],
        ui_components['freeze_backbone'],
        ui_components['freeze_layers']
    ]
    
    ui_components['form'] = widgets.VBox(
        ui_components['form_items'],
        layout=widgets.Layout(padding='10px')
    )
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['backbone_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    ui_components['buttons'] = widgets.HBox(
        [ui_components['save_button'], ui_components['reset_button']],
        layout=widgets.Layout(padding='10px')
    )
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        ui_components['title'],
        ui_components['form'],
        ui_components['info_box'],
        ui_components['buttons'],
        ui_components['status']
    ])
    
    return ui_components
