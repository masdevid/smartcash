"""
File: smartcash/ui/training_config/training_strategy/components/training_strategy_components.py
Deskripsi: Komponen UI untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, create_divider
from smartcash.ui.utils.header_utils import create_header

def create_training_strategy_ui_components() -> Dict[str, Any]:
    """
    Membuat komponen UI untuk konfigurasi strategi pelatihan model.
    
    Returns:
        Dict berisi komponen UI
    """
    # Inisialisasi komponen
    ui_components = {}
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value=f"<h3>{ICONS.get('training', '🏋️')} Konfigurasi Strategi Pelatihan</h3>"
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
    
    ui_components['layer_mode'] = widgets.RadioButtons(
        options=['single', 'multilayer'],
        value='single',
        description='Layer mode:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
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
    
    # Informasi strategi pelatihan
    ui_components['training_strategy_info'] = widgets.HTML(
        value="<p style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'><b>ℹ️ Info:</b> Konfigurasi strategi pelatihan dasar untuk model YOLOv5 dengan EfficientNet backbone.</p>"
    )
    
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
        ui_components['mixed_precision'],
        ui_components['layer_mode']
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
    ui_components['tabs'].set_title(0, f"{ICONS.get('settings', '⚙️')} Utilitas Training")
    ui_components['tabs'].set_title(1, f"{ICONS.get('check', '✓')} Validasi")
    ui_components['tabs'].set_title(2, f"{ICONS.get('scale', '📏')} Multi-scale")
    
    ui_components['info_box'] = widgets.VBox(
        [ui_components['training_strategy_info']],
        layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0')
    )
    
    # Tambahkan tombol konfigurasi
    from smartcash.ui.components.config_buttons import create_config_buttons
    config_buttons = create_config_buttons()
    ui_components.update({
        'save_button': config_buttons['save_button'],
        'reset_button': config_buttons['reset_button'],
        'config_buttons': config_buttons['container']
    })
    
    # Header dengan komponen standar
    header = create_header(
        title=f"{ICONS.get('training', '🏋️')} Training Strategy Configuration",
        description="Konfigurasi strategi pelatihan untuk model deteksi mata uang"
    )
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS.get('info', 'ℹ️')} Konfigurasi strategi pelatihan model</p>
        </div>"""
    )
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[ui_components['status']], selected_index=None)
    log_accordion.set_title(0, f"{ICONS.get('file', '📄')} Training Strategy Logs")
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('settings', '⚙️')} Training Strategy</h4>"),
        ui_components['tabs'],
        ui_components['info_box'],
        create_divider(),
        config_buttons['container'],
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
