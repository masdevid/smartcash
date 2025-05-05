"""
File: smartcash/ui/training_config/hyperparameters/components/hyperparameters_components.py
Deskripsi: Komponen UI untuk konfigurasi hyperparameter model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def create_hyperparameters_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk pengaturan hyperparameter.
    
    Args:
        config: Konfigurasi training
        
    Returns:
        Dict berisi komponen UI
    """
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, create_divider
    
    # Inisialisasi komponen
    ui_components = {}
    
    # Tambahkan komponen status
    ui_components['status'] = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value="<h3>🔢 Konfigurasi Hyperparameter</h3>"
    )
    
    # Tab untuk kategori hyperparameter
    ui_components['tabs'] = widgets.Tab()
    
    # Tab 1: Optimizer
    ui_components['optimizer_type'] = widgets.Dropdown(
        options=['SGD', 'Adam', 'AdamW', 'RMSprop'],
        value='AdamW',
        description='Optimizer:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['learning_rate'] = widgets.FloatLogSlider(
        value=0.001,
        base=10,
        min=-5,  # 10^-5 = 0.00001
        max=-1,  # 10^-1 = 0.1
        step=0.1,
        description='Learning rate:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['weight_decay'] = widgets.FloatLogSlider(
        value=0.0005,
        base=10,
        min=-6,  # 10^-6 = 0.000001
        max=-2,  # 10^-2 = 0.01
        step=0.1,
        description='Weight decay:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['momentum'] = widgets.FloatSlider(
        value=0.9,
        min=0.0,
        max=0.99,
        step=0.01,
        description='Momentum:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 2: Scheduler
    ui_components['scheduler_type'] = widgets.Dropdown(
        options=['step', 'cosine', 'plateau', 'none'],
        value='cosine',
        description='Scheduler:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['warmup_epochs'] = widgets.IntSlider(
        value=3,
        min=0,
        max=10,
        step=1,
        description='Warmup epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['step_size'] = widgets.IntSlider(
        value=30,
        min=1,
        max=100,
        step=1,
        description='Step size:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['gamma'] = widgets.FloatSlider(
        value=0.1,
        min=0.01,
        max=0.5,
        step=0.01,
        description='Gamma:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 3: Augmentasi
    ui_components['use_augmentation'] = widgets.Checkbox(
        value=True,
        description='Gunakan augmentasi data',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['mosaic'] = widgets.Checkbox(
        value=True,
        description='Mosaic augmentation',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['mixup'] = widgets.Checkbox(
        value=False,
        description='Mixup augmentation',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['flip'] = widgets.Checkbox(
        value=True,
        description='Random flip',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['hsv_h'] = widgets.FloatSlider(
        value=0.015,
        min=0.0,
        max=0.1,
        step=0.001,
        description='HSV hue:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['hsv_s'] = widgets.FloatSlider(
        value=0.7,
        min=0.0,
        max=1.0,
        step=0.01,
        description='HSV saturation:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['hsv_v'] = widgets.FloatSlider(
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.01,
        description='HSV value:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Informasi hyperparameter
    ui_components['hyperparameters_info'] = widgets.HTML(
        value="<p>Informasi hyperparameter akan ditampilkan di sini</p>"
    )
    
    # Tombol aksi akan ditambahkan dari initializer
    
    # Susun layout
    optimizer_box = widgets.VBox([
        ui_components['optimizer_type'],
        ui_components['learning_rate'],
        ui_components['weight_decay'],
        ui_components['momentum']
    ])
    
    scheduler_box = widgets.VBox([
        ui_components['scheduler_type'],
        ui_components['warmup_epochs'],
        ui_components['step_size'],
        ui_components['gamma']
    ])
    
    augmentation_box = widgets.VBox([
        ui_components['use_augmentation'],
        ui_components['mosaic'],
        ui_components['mixup'],
        ui_components['flip'],
        ui_components['hsv_h'],
        ui_components['hsv_s'],
        ui_components['hsv_v']
    ])
    
    # Buat tabs
    ui_components['tabs'].children = [optimizer_box, scheduler_box, augmentation_box]
    ui_components['tabs'].set_title(0, 'Optimizer')
    ui_components['tabs'].set_title(1, 'Scheduler')
    ui_components['tabs'].set_title(2, 'Augmentasi')
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['hyperparameters_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    # Placeholder untuk tombol konfigurasi yang akan ditambahkan dari initializer
    ui_components['buttons_placeholder'] = widgets.HBox(
        [],
        layout=widgets.Layout(padding='10px')
    )
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['settings']} Hyperparameters Configuration", 
                          "Konfigurasi hyperparameter untuk training model deteksi mata uang")
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Konfigurasi hyperparameter training</p>
        </div>"""
    )
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[ui_components['status']], selected_index=0)
    log_accordion.set_title(0, f"{ICONS['file']} Hyperparameters Logs")
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Training Parameters</h4>"),
        ui_components['tabs'],
        create_divider(),
        ui_components['buttons_placeholder'],
        log_accordion
    ])
    
    # Tambahkan referensi komponen tambahan ke ui_components
    ui_components.update({
        'header': header,
        'status_panel': status_panel,
        'log_accordion': log_accordion,
        'module_name': 'hyperparameters'
    })
    
    return ui_components
