"""
File: smartcash/ui/training/components/training_components.py
Deskripsi: Komponen UI untuk proses training model SmartCash
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, Optional

def create_training_components() -> Dict[str, Any]:
    """
    Membuat komponen UI untuk proses training model.
    
    Returns:
        Dict berisi komponen UI untuk training
    """
    # Komponen status dan informasi
    status_panel = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            padding='10px',
            margin='5px 0'
        )
    )
    
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
    
    # Komponen utama
    main_box = widgets.VBox([
        info_box,
        status_label,
        progress_bar,
        control_box,
        options_box,
        metrics_box,
        status_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Kumpulkan semua komponen
    components = {
        'main_box': main_box,
        'info_box': info_box,
        'status_panel': status_panel,
        'progress_bar': progress_bar,
        'status_label': status_label,
        'start_button': start_button,
        'stop_button': stop_button,
        'save_checkpoints': save_checkpoints,
        'use_tensorboard': use_tensorboard,
        'metrics_box': metrics_box
    }
    
    return components
