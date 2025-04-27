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
    
    # Import ModelManager untuk mendapatkan model yang dioptimalkan
    from smartcash.model.manager import ModelManager
    from smartcash.model.config.backbone_config import BackboneConfig
    
    # Daftar backbone yang didukung
    backbone_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
    
    # Daftar model yang dioptimalkan
    optimized_models = ModelManager.OPTIMIZED_MODELS
    
    # Import konstanta untuk styling yang konsisten
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value=f"<h3>{ICONS['model']} Konfigurasi Backbone Model</h3>"
    )
    
    # Dropdown untuk memilih model yang dioptimalkan
    model_options = [(f"{key}: {value['description']}", key) for key, value in optimized_models.items()]
    
    ui_components['model_type'] = widgets.Dropdown(
        options=model_options,
        value='efficient_optimized',
        description='Model:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='500px')
    )
    
    # Dropdown backbone akan otomatis diupdate berdasarkan model yang dipilih
    ui_components['backbone_type'] = widgets.Dropdown(
        options=backbone_options,
        value=optimized_models['efficient_optimized']['backbone'],
        description='Backbone:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
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
    
    # Fitur optimasi
    ui_components['use_attention'] = widgets.Checkbox(
        value=optimized_models['efficient_optimized']['use_attention'],
        description='Gunakan FeatureAdapter (Attention)',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
    )
    
    ui_components['use_residual'] = widgets.Checkbox(
        value=optimized_models['efficient_optimized']['use_residual'],
        description='Gunakan ResidualAdapter',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
    )
    
    ui_components['use_ciou'] = widgets.Checkbox(
        value=optimized_models['efficient_optimized']['use_ciou'],
        description='Gunakan CIoU Loss',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
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
    
    # Susun layout dengan cards
    
    # Card untuk model selection
    model_selection_card = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['model']} Pilihan Model</h4>"),
        ui_components['model_type']
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='5px',
        padding='15px',
        margin='0 0 15px 0',
        background_color='#f8f9fa'
    ))
    
    # Card untuk konfigurasi backbone
    backbone_config_card = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['settings']} Konfigurasi Backbone</h4>"),
        ui_components['backbone_type'],
        ui_components['pretrained'],
        ui_components['freeze_backbone'],
        ui_components['freeze_layers']
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='5px',
        padding='15px',
        margin='0 0 15px 0',
        background_color='#f8f9fa'
    ))
    
    # Card untuk fitur optimasi
    optimization_card = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['optimization']} Fitur Optimasi</h4>"),
        ui_components['use_attention'],
        ui_components['use_residual'],
        ui_components['use_ciou']
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='5px',
        padding='15px',
        margin='0 0 15px 0',
        background_color='#f8f9fa'
    ))
    
    # Kolom kiri: Model selection dan backbone config
    left_column = widgets.VBox([
        model_selection_card,
        backbone_config_card
    ])
    
    # Buat info_box terlebih dahulu
    ui_components['info_box'] = widgets.VBox(
        [widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['info']} Informasi Model</h4>"),
         ui_components['backbone_info']],
        layout=widgets.Layout(
            padding='15px', 
            border='1px solid #ddd', 
            border_radius='5px',
            margin='0 0 15px 0',
            background_color='#f8f9fa'
        )
    )
    
    # Kolom kanan: Fitur optimasi dan info backbone
    right_column = widgets.VBox([
        optimization_card,
        ui_components['info_box']
    ])
    
    # Gabungkan dalam layout grid
    ui_components['form'] = widgets.HBox([
        left_column,
        right_column
    ], layout=widgets.Layout(padding='10px'))
    
    # Info box sudah dibuat sebelumnya
    
    ui_components['buttons'] = widgets.HBox(
        [ui_components['save_button'], ui_components['reset_button']],
        layout=widgets.Layout(padding='10px')
    )
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        ui_components['title'],
        ui_components['form'],
        ui_components['buttons'],
        ui_components['status']
    ])
    
    # Handler untuk mengupdate UI berdasarkan model yang dipilih
    def on_model_change(change):
        model_key = change['new']
        model_config = optimized_models[model_key]
        
        # Update backbone
        ui_components['backbone_type'].value = model_config['backbone']
        
        # Update fitur optimasi
        ui_components['use_attention'].value = model_config.get('use_attention', False)
        ui_components['use_residual'].value = model_config.get('use_residual', False)
        ui_components['use_ciou'].value = model_config.get('use_ciou', False)
        
        # Update informasi backbone
        backbone_info = f"""
        <div style='padding: 10px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;'>
            <h4>{model_key.replace('_', ' ').title()}</h4>
            <p><strong>Deskripsi:</strong> {model_config['description']}</p>
            <p><strong>Backbone:</strong> {model_config['backbone']}</p>
            <p><strong>Fitur Optimasi:</strong></p>
            <ul>
                <li>FeatureAdapter (Attention): {'✅ Aktif' if model_config.get('use_attention', False) else '❌ Tidak aktif'}</li>
                <li>ResidualAdapter: {'✅ Aktif' if model_config.get('use_residual', False) else '❌ Tidak aktif'}</li>
                <li>CIoU Loss: {'✅ Aktif' if model_config.get('use_ciou', False) else '❌ Tidak aktif'}</li>
            </ul>
        </div>
        """
        ui_components['backbone_info'].value = backbone_info
    
    # Daftarkan handler
    ui_components['model_type'].observe(on_model_change, names='value')
    
    # Trigger handler untuk inisialisasi awal
    on_model_change({'new': ui_components['model_type'].value})
    
    return ui_components
