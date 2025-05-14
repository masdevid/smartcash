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
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, create_divider
    
    # Inisialisasi komponen
    ui_components = {}
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value="<h3>🏋️ Konfigurasi Strategi Pelatihan</h3>"
    )
    
    # Tab untuk kategori strategi pelatihan
    ui_components['tabs'] = widgets.Tab()
    
    # Tab 1: Parameter Utilitas Training
    ui_components['experiment_name'] = widgets.Text(
        value='efficientnet_b4_training',
        description='Experiment name:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='500px')
    )
    
    ui_components['checkpoint_dir'] = widgets.Text(
        value='/content/runs/train/checkpoints',
        description='Checkpoint dir:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='500px')
    )
    
    ui_components['tensorboard'] = widgets.Checkbox(
        value=True,
        description='Enable TensorBoard',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['log_metrics_every'] = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Log metrics every:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['visualize_batch_every'] = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description='Visualize batch every:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['gradient_clipping'] = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Gradient clipping:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['mixed_precision'] = widgets.Checkbox(
        value=True,
        description='Enable mixed precision',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    # Tab 2: Validasi dan Evaluasi
    ui_components['validation_frequency'] = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        step=1,
        description='Validation frequency:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['iou_threshold'] = widgets.FloatSlider(
        value=0.6,
        min=0.1,
        max=0.9,
        step=0.05,
        description='IoU threshold:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    ui_components['conf_threshold'] = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.0001,
        description='Conf threshold:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Tab 3: Multi-scale Training
    ui_components['multi_scale'] = widgets.Checkbox(
        value=True,
        description='Enable multi-scale training',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    # Menghapus tab eksperimen dan multi-GPU sesuai permintaan
    
    # Informasi strategi pelatihan
    ui_components['training_strategy_info'] = widgets.HTML(
        value="<p style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'><b>ℹ️ Info:</b> Konfigurasi strategi pelatihan dasar untuk model YOLOv5 dengan EfficientNet backbone.</p>"
    )
    
    # Tombol aksi akan ditambahkan dari initializer
    
    # Status indicator
    ui_components['status'] = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Susun layout
    utils_box = widgets.VBox([
        ui_components['experiment_name'],
        ui_components['checkpoint_dir'],
        ui_components['tensorboard'],
        ui_components['log_metrics_every'],
        ui_components['visualize_batch_every'],
        ui_components['gradient_clipping'],
        ui_components['mixed_precision']
    ])
    
    validation_box = widgets.VBox([
        ui_components['validation_frequency'],
        ui_components['iou_threshold'],
        ui_components['conf_threshold']
    ])
    
    multiscale_box = widgets.VBox([
        ui_components['multi_scale']
    ])
    
    # Buat tabs
    ui_components['tabs'].children = [utils_box, validation_box, multiscale_box]
    ui_components['tabs'].set_title(0, 'Utilitas Training')
    ui_components['tabs'].set_title(1, 'Validasi')
    ui_components['tabs'].set_title(2, 'Multi-scale')
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['training_strategy_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    # Placeholder untuk tombol konfigurasi yang akan ditambahkan dari initializer
    ui_components['buttons_placeholder'] = widgets.HBox(
        [],
        layout=widgets.Layout(padding='10px')
    )
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['training']} Training Strategy Configuration", 
                          "Konfigurasi strategi pelatihan untuk model deteksi mata uang")
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Konfigurasi strategi pelatihan model</p>
        </div>"""
    )
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[ui_components['status']], selected_index=0)
    log_accordion.set_title(0, f"{ICONS['file']} Training Strategy Logs")
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Training Strategy</h4>"),
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
        'module_name': 'training_strategy'
    })
    
    return ui_components
