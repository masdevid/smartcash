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
    ui_components['val_split'] = widgets.FloatSlider(
        value=0.2,
        min=0.1,
        max=0.5,
        step=0.05,
        description='Val split:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
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
    
    # Tab 3: Eksperimen
    ui_components['experiment_name'] = widgets.Text(
        value='yolov5_efficientnet_b4',
        description='Nama eksperimen:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['save_period'] = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Save period:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['resume'] = widgets.Checkbox(
        value=False,
        description='Resume training',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['checkpoint_path'] = widgets.Text(
        value='',
        placeholder='Path ke checkpoint (opsional)',
        description='Checkpoint:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px'),
        disabled=True
    )
    
    # Tab 4: Multi-GPU
    ui_components['use_multi_gpu'] = widgets.Checkbox(
        value=False,
        description='Gunakan multi-GPU',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['sync_bn'] = widgets.Checkbox(
        value=True,
        description='Sync BatchNorm',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px'),
        disabled=True
    )
    
    ui_components['distributed'] = widgets.Checkbox(
        value=False,
        description='Distributed training',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px'),
        disabled=True
    )
    
    # Informasi strategi pelatihan
    ui_components['training_strategy_info'] = widgets.HTML(
        value="<p>Informasi strategi pelatihan akan ditampilkan di sini</p>"
    )
    
    # Buat tombol aksi
    ui_components['save_button'] = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='success',
        icon='save',
        tooltip='Simpan konfigurasi strategi pelatihan',
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
    
    experiment_box = widgets.VBox([
        ui_components['experiment_name'],
        ui_components['save_period'],
        ui_components['resume'],
        ui_components['checkpoint_path']
    ])
    
    multi_gpu_box = widgets.VBox([
        ui_components['use_multi_gpu'],
        ui_components['sync_bn'],
        ui_components['distributed']
    ])
    
    # Buat tabs
    ui_components['tabs'].children = [basic_box, validation_box, experiment_box, multi_gpu_box]
    ui_components['tabs'].set_title(0, 'Parameter Dasar')
    ui_components['tabs'].set_title(1, 'Validasi')
    ui_components['tabs'].set_title(2, 'Eksperimen')
    ui_components['tabs'].set_title(3, 'Multi-GPU')
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['training_strategy_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    ui_components['buttons'] = widgets.HBox(
        [ui_components['save_button'], ui_components['reset_button']],
        layout=widgets.Layout(padding='10px')
    )
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        ui_components['title'],
        ui_components['tabs'],
        ui_components['info_box'],
        ui_components['buttons'],
        ui_components['status']
    ])
    
    return ui_components
