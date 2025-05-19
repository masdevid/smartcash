"""
File: smartcash/ui/training/components/training_components.py
Deskripsi: Komponen UI untuk proses training model SmartCash
"""

import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import io
import base64

def create_training_components() -> Dict[str, Any]:
    """
    Membuat komponen UI untuk proses training model.
    
    Returns:
        Dict berisi komponen UI untuk training
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import create_divider
    
    # Import shared components
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    
    # Header
    header = create_header(
        f"{ICONS.get('training', '🚀')} Model Training", 
        "Latih model dengan konfigurasi dari file YAML"
    )
    
    # Status panel untuk log
    status_panel = create_status_panel("Konfigurasi training model", "info")
    
    # Informasi model dan konfigurasi
    info_box = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            padding='10px',
            margin='5px 0',
            max_height='200px',
            overflow_y='auto'
        )
    )
    
    # Chart untuk visualisasi metrik realtime
    chart_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            height='300px',
            margin='10px 0'
        )
    )
    
    # Checkbox untuk opsi tambahan
    save_checkpoints = widgets.Checkbox(
        value=True,
        description='Simpan checkpoint',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    use_tensorboard = widgets.Checkbox(
        value=True,
        description='Gunakan TensorBoard',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Dropdown untuk backbone model
    backbone_dropdown = widgets.Dropdown(
        options=['EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4'],
        value='EfficientNet-B4',
        description='Backbone:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )
    
    # Input untuk epochs
    epochs_input = widgets.IntText(
        value=100,
        description='Epochs:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )
    
    # Input untuk batch size
    batch_size_input = widgets.IntText(
        value=16,
        description='Batch Size:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )
    
    # Layout komponen opsi
    options_box = widgets.HBox([
        save_checkpoints,
        use_tensorboard
    ], layout=widgets.Layout(margin='10px 0'))
    
    # Layout komponen konfigurasi
    config_box = widgets.VBox([
        backbone_dropdown,
        epochs_input,
        batch_size_input,
        options_box
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Box untuk metrik training
    metrics_box = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            padding='10px',
            margin='5px 0',
            max_height='300px',
            overflow_y='auto'
        )
    )
    
    # Fungsi untuk membuat chart metrik realtime
    def create_metrics_chart():
        plt.figure(figsize=(10, 6))
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tambahkan contoh plot dummy dengan label untuk menghindari warning
        epochs = [0]
        # Hanya tambahkan plot dengan label jika chart kosong (inisialisasi awal)
        plt.plot(epochs, [0], '-', alpha=0, label='Train Loss')
        plt.plot(epochs, [0], '--', alpha=0, label='Val Loss')
        plt.plot(epochs, [0], ':', alpha=0, label='mAP')
        
        # Tampilkan legend dengan lokasi yang optimal
        plt.legend(loc='best')
        
        # Konversi plot ke gambar untuk ditampilkan di widget
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return widgets.HTML(f'<img src="data:image/png;base64,{img_str}" width="100%">')
    
    # Inisialisasi chart kosong
    with chart_output:
        display(create_metrics_chart())
        
    # Buat tombol-tombol training dengan shared component
    action_buttons = create_action_buttons(
        primary_label="Mulai Training",
        primary_icon="play",
        secondary_buttons=[
            ("Hentikan", "stop", "danger"),
            ("Reset", "refresh", "warning")
        ],
        cleanup_enabled=True
    )
    
    # Progress tracking dengan shared component
    progress_components = create_progress_tracking(
        module_name='training',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )
    
    # Log accordion dengan shared component
    log_components = create_log_accordion(
        module_name='training',
        height='200px',
        width='100%'
    )
    
    # Komponen utama
    main_box = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('settings', '⚙️')} Konfigurasi Training</h4>"),
        config_box,
        create_divider(),
        action_buttons['container'],
        progress_components['progress_container'],
        chart_output,
        metrics_box,
        log_components['log_accordion']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Kumpulkan semua komponen
    components = {
        'ui': main_box,
        'header': header,
        'status_panel': status_panel,
        'info_box': info_box,
        'backbone_dropdown': backbone_dropdown,
        'epochs_input': epochs_input,
        'batch_size_input': batch_size_input,
        'save_checkpoints': save_checkpoints,
        'use_tensorboard': use_tensorboard,
        'metrics_box': metrics_box,
        'chart_output': chart_output,
        'create_metrics_chart': create_metrics_chart,
        'module_name': 'training'
    }
    
    # Tambahkan komponen action buttons
    components.update({
        'start_button': action_buttons['primary_button'],
        'stop_button': action_buttons['secondary_buttons'][0] if 'secondary_buttons' in action_buttons else None,
        'reset_button': action_buttons['secondary_buttons'][1] if 'secondary_buttons' in action_buttons else action_buttons.get('reset_button'),
        'cleanup_button': action_buttons.get('cleanup_button'),
        'button_container': action_buttons['container']
    })
    
    # Tambahkan komponen progress tracking
    components.update({
        'progress_bar': progress_components['progress_bar'],
        'progress_container': progress_components['progress_container'],
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label')
    })
    
    # Tambahkan komponen log
    components.update({
        'status': log_components['log_output'],
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion']
    })
    
    return components
