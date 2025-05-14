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
    from smartcash.ui.utils.layout_utils import MAIN_CONTAINER, OUTPUT_WIDGET, BUTTON
    from smartcash.ui.components.status_panel import create_status_panel
    
    # Header
    header = create_header(
        "🚀 Model Training", 
        "Latih model dengan konfigurasi dari file YAML"
    )
    
    # Status panel untuk log
    status_panel, ui_log = create_status_panel()
    
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
    
    # Progress bar untuk training
    progress_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Label untuk status training
    status_label = widgets.HTML(
        value='<span style="color:#888">Siap untuk memulai training...</span>',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Tombol untuk kontrol training
    start_button = widgets.Button(
        description='Mulai Training',
        button_style='success',
        icon='play',
        tooltip='Mulai proses training model',
        layout=widgets.Layout(width='150px')
    )
    
    stop_button = widgets.Button(
        description='Hentikan',
        button_style='danger',
        icon='stop',
        disabled=True,
        tooltip='Hentikan proses training',
        layout=widgets.Layout(width='150px')
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
    
    # Layout komponen
    control_box = widgets.HBox([
        start_button, 
        stop_button
    ], layout=widgets.Layout(margin='10px 0'))
    
    options_box = widgets.HBox([
        save_checkpoints,
        use_tensorboard
    ], layout=widgets.Layout(margin='10px 0'))
    
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
        plt.legend()
        
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
    
    # Komponen utama
    main_box = widgets.VBox([
        header,
        info_box,
        status_label,
        progress_bar,
        control_box,
        options_box,
        chart_output,
        metrics_box,
        status_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Kumpulkan semua komponen
    components = {
        'main_box': main_box,
        'header': header,
        'info_box': info_box,
        'status_panel': status_panel,
        'ui_log': ui_log,
        'progress_bar': progress_bar,
        'status_label': status_label,
        'start_button': start_button,
        'stop_button': stop_button,
        'save_checkpoints': save_checkpoints,
        'use_tensorboard': use_tensorboard,
        'metrics_box': metrics_box,
        'chart_output': chart_output,
        'create_metrics_chart': create_metrics_chart
    }
    
    return components
