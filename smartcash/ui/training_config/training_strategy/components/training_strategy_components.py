"""
File: smartcash/ui/training_config/training_strategy/components/training_strategy_components.py
Deskripsi: Komponen UI untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def create_training_strategy_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk konfigurasi strategi pelatihan model.
    
    Args:
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI
    """
    # Inisialisasi komponen
    ui_components = {}
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value="<h3>🏋️ Konfigurasi Strategi Pelatihan</h3>"
    )
    
    # Tab untuk kategori strategi pelatihan
    ui_components['tabs'] = widgets.Tab()
    
    # Tab 1: Parameter Dasar
    ui_components['batch_size'] = widgets.IntSlider(
        value=16,
        min=1,
        max=64,
        step=1,
        description='Batch size:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['epochs'] = widgets.IntSlider(
        value=100,
        min=1,
        max=300,
        step=1,
        description='Epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['image_size'] = widgets.IntSlider(
        value=640,
        min=320,
        max=1280,
        step=32,
        description='Image size:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['workers'] = widgets.IntSlider(
        value=4,
        min=0,
        max=16,
        step=1,
        description='Workers:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 2: Validasi dan Evaluasi
    
    # Tambahkan val_split dengan nilai default 15% (0.15)
    ui_components['val_split'] = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.01,
        description='Val split:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px'),
        readout_format='.0%'
    )
    
    ui_components['val_frequency'] = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        step=1,
        description='Val frequency:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['early_stopping'] = widgets.Checkbox(
        value=True,
        description='Early stopping',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['patience'] = widgets.IntSlider(
        value=10,
        min=1,
        max=30,
        step=1,
        description='Patience:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Menghapus tab eksperimen dan multi-GPU sesuai permintaan
    
    # Informasi strategi pelatihan
    ui_components['training_strategy_info'] = widgets.HTML(
        value="<p style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'><b>ℹ️ Info:</b> Konfigurasi strategi pelatihan dasar untuk model YOLOv5 dengan EfficientNet backbone.</p>"
    )
    
    # Tombol aksi akan ditambahkan dari initializer
    
    # Status indicator
    ui_components['status'] = widgets.Output(
        layout=widgets.Layout(width='100%', padding='10px')
    )
    
    # Susun layout
    basic_box = widgets.VBox([
        ui_components['batch_size'],
        ui_components['epochs'],
        ui_components['image_size'],
        ui_components['workers']
    ])
    
    validation_box = widgets.VBox([
        ui_components['val_split'],
        ui_components['val_frequency'],
        ui_components['early_stopping'],
        ui_components['patience']
    ])
    
    # Buat tabs (hanya parameter dasar dan validasi)
    ui_components['tabs'].children = [basic_box, validation_box]
    ui_components['tabs'].set_title(0, 'Parameter Dasar')
    ui_components['tabs'].set_title(1, 'Validasi')
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['training_strategy_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    # Placeholder untuk tombol konfigurasi yang akan ditambahkan dari initializer
    ui_components['buttons_placeholder'] = widgets.HBox(
        [],
        layout=widgets.Layout(padding='10px')
    )
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        ui_components['title'],
        ui_components['tabs'],
        ui_components['info_box'],
        ui_components['buttons_placeholder'],
        ui_components['status']
    ])
    
    return ui_components
