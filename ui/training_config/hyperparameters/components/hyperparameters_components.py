"""
File: smartcash/ui/training_config/hyperparameters/components/hyperparameters_components.py
Deskripsi: Komponen UI untuk konfigurasi hyperparameter model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def create_hyperparameters_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk pengaturan hyperparameter berdasarkan hyperparameter_config.yaml.
    
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
    
    # Tab 1: Parameter dasar
    ui_components['batch_size'] = widgets.IntSlider(
        value=16,
        min=1,
        max=64,
        step=1,
        description='Batch size:',
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
    
    ui_components['epochs'] = widgets.IntSlider(
        value=100,
        min=10,
        max=300,
        step=10,
        description='Epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 2: Optimizer
    ui_components['optimizer_type'] = widgets.Dropdown(
        options=['SGD', 'Adam', 'AdamW', 'RMSprop'],
        value='Adam',  # Sesuai dengan hyperparameter_config.yaml
        description='Optimizer:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['learning_rate'] = widgets.FloatLogSlider(
        value=0.001,  # Sesuai dengan hyperparameter_config.yaml
        base=10,
        min=-5,  # 10^-5 = 0.00001
        max=-1,  # 10^-1 = 0.1
        step=0.1,
        description='Learning rate:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['weight_decay'] = widgets.FloatLogSlider(
        value=0.0005,  # Sesuai dengan hyperparameter_config.yaml
        base=10,
        min=-6,  # 10^-6 = 0.000001
        max=-2,  # 10^-2 = 0.01
        step=0.1,
        description='Weight decay:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['momentum'] = widgets.FloatSlider(
        value=0.937,  # Sesuai dengan hyperparameter_config.yaml
        min=0.0,
        max=0.99,
        step=0.001,
        description='Momentum:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 3: Scheduler
    ui_components['lr_scheduler'] = widgets.Dropdown(
        options=['step', 'cosine', 'plateau', 'none'],
        value='cosine',  # Sesuai dengan hyperparameter_config.yaml
        description='Scheduler:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['warmup_epochs'] = widgets.IntSlider(
        value=3,  # Sesuai dengan hyperparameter_config.yaml
        min=0,
        max=10,
        step=1,
        description='Warmup epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['warmup_momentum'] = widgets.FloatSlider(
        value=0.8,  # Sesuai dengan hyperparameter_config.yaml
        min=0.0,
        max=0.99,
        step=0.01,
        description='Warmup momentum:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['warmup_bias_lr'] = widgets.FloatSlider(
        value=0.1,  # Sesuai dengan hyperparameter_config.yaml
        min=0.01,
        max=1.0,
        step=0.01,
        description='Warmup bias LR:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 4: Regularisasi
    ui_components['augment'] = widgets.Checkbox(
        value=True,  # Sesuai dengan hyperparameter_config.yaml
        description='Gunakan augmentasi data',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['dropout'] = widgets.FloatSlider(
        value=0.0,  # Sesuai dengan hyperparameter_config.yaml
        min=0.0,
        max=0.5,
        step=0.01,
        description='Dropout rate:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 5: Loss
    ui_components['box_loss_gain'] = widgets.FloatSlider(
        value=0.05,  # Sesuai dengan hyperparameter_config.yaml
        min=0.01,
        max=0.1,
        step=0.01,
        description='Box loss gain:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['cls_loss_gain'] = widgets.FloatSlider(
        value=0.5,  # Sesuai dengan hyperparameter_config.yaml
        min=0.1,
        max=1.0,
        step=0.1,
        description='Class loss gain:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['obj_loss_gain'] = widgets.FloatSlider(
        value=1.0,  # Sesuai dengan hyperparameter_config.yaml
        min=0.1,
        max=2.0,
        step=0.1,
        description='Object loss gain:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 6: Early Stopping & Checkpoint
    ui_components['early_stopping_enabled'] = widgets.Checkbox(
        value=True,  # Sesuai dengan hyperparameter_config.yaml
        description='Early stopping',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['early_stopping_patience'] = widgets.IntSlider(
        value=15,  # Sesuai dengan hyperparameter_config.yaml
        min=1,
        max=30,
        step=1,
        description='Patience:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['early_stopping_min_delta'] = widgets.FloatLogSlider(
        value=0.001,  # Sesuai dengan hyperparameter_config.yaml
        base=10,
        min=-5,  # 10^-5 = 0.00001
        max=-2,  # 10^-2 = 0.01
        step=0.1,
        description='Min delta:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['checkpoint_save_best'] = widgets.Checkbox(
        value=True,  # Sesuai dengan hyperparameter_config.yaml
        description='Save best model',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['checkpoint_save_period'] = widgets.IntSlider(
        value=10,  # Sesuai dengan hyperparameter_config.yaml
        min=1,
        max=50,
        step=1,
        description='Save period:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Informasi hyperparameter
    ui_components['hyperparameters_info'] = widgets.HTML(
        value="<p>Informasi hyperparameter akan ditampilkan di sini</p>"
    )
    
    # Tombol aksi akan ditambahkan dari initializer
    
    # Buat tab
    basic_tab = widgets.VBox([
        ui_components['batch_size'],
        ui_components['image_size'],
        ui_components['epochs']
    ])
    
    optimizer_tab = widgets.VBox([
        ui_components['optimizer_type'],
        ui_components['learning_rate'],
        ui_components['weight_decay'],
        ui_components['momentum']
    ])
    
    scheduler_tab = widgets.VBox([
        ui_components['lr_scheduler'],
        ui_components['warmup_epochs'],
        ui_components['warmup_momentum'],
        ui_components['warmup_bias_lr']
    ])
    
    regularization_tab = widgets.VBox([
        ui_components['augment'],
        ui_components['dropout']
    ])
    
    loss_tab = widgets.VBox([
        ui_components['box_loss_gain'],
        ui_components['cls_loss_gain'],
        ui_components['obj_loss_gain']
    ])
    
    early_stopping_tab = widgets.VBox([
        ui_components['early_stopping_enabled'],
        ui_components['early_stopping_patience'],
        ui_components['early_stopping_min_delta'],
        ui_components['checkpoint_save_best'],
        ui_components['checkpoint_save_period']
    ])
    
    # Buat tabs
    ui_components['tabs'].children = [basic_tab, optimizer_tab, scheduler_tab, regularization_tab, loss_tab, early_stopping_tab]
    ui_components['tabs'].set_title(0, 'Parameter Dasar')
    ui_components['tabs'].set_title(1, 'Optimizer')
    ui_components['tabs'].set_title(2, 'Scheduler')
    ui_components['tabs'].set_title(3, 'Regularisasi')
    ui_components['tabs'].set_title(4, 'Loss')
    ui_components['tabs'].set_title(5, 'Early Stopping & Checkpoint')
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['hyperparameters_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    # Placeholder untuk tombol konfigurasi yang akan ditambahkan dari initializer
    ui_components['buttons_placeholder'] = widgets.HBox(
        [],
        layout=widgets.Layout(padding='10px')
    )
    
    # Tambahkan keterangan sinkronisasi otomatis
    ui_components['sync_info'] = widgets.HTML(
        value=f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('info', 'ℹ️')} Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.</div>"
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
