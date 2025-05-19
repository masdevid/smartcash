"""
File: smartcash/ui/training/components/training_component.py
Deskripsi: Komponen UI utama untuk training model menggunakan shared components
"""

import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import io
import base64

def create_training_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk proses training model.
    
    Args:
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI untuk training
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import create_divider
    from smartcash.common.logger import get_logger
    
    # Import shared components
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.feature_checkbox_group import create_feature_checkbox_group
    from smartcash.ui.components.config_form import create_config_form
    from smartcash.ui.info_boxes.training_info import get_training_info
    
    # Dapatkan logger
    logger = get_logger("training_ui")
    
    # Header
    header = create_header(
        f"{ICONS.get('training', '🚀')} Model Training", 
        "Latih model dengan konfigurasi yang telah diatur sebelumnya"
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
    
    # Chart untuk visualisasi metrik realtime dengan layout yang lebih baik
    chart_output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            height='300px',
            margin='20px 0',
            border='1px solid #ddd',
            padding='10px'
        )
    )
    
    # Buat tombol-tombol untuk training dengan shared component
    action_buttons = create_action_buttons(
        primary_label="Mulai Training",
        primary_icon="play",
        cleanup_enabled=True,
        layout=widgets.Layout(width="100%", margin="10px 0", gap="5px")
    )
    
    # Sesuaikan label dan tooltip tombol
    action_buttons['primary_button'].tooltip = "Mulai proses training model dengan konfigurasi yang sudah diatur"
    action_buttons['stop_button'].description = "Hentikan"
    action_buttons['stop_button'].tooltip = "Hentikan proses training"
    action_buttons['cleanup_button'].description = "Bersihkan"
    action_buttons['cleanup_button'].tooltip = "Bersihkan hasil training"
    action_buttons['stop_button'].icon = "stop"
    action_buttons['cleanup_button'].icon = "trash"
    
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
    
    # Help panel dengan komponen info_box standar
    help_panel = get_training_info() if 'get_training_info' in globals() else widgets.HTML(
        f"<div style='padding: 10px; background-color: #f8f9fa; border-left: 5px solid {COLORS.get('info', '#3498db')}; margin-top: 15px;'>"
        f"<h4>{ICONS.get('info', 'ℹ️')} Informasi Training</h4>"
        "<p>Training model akan menggunakan konfigurasi yang telah diatur pada modul-modul sebelumnya.</p>"
        "<p>Pastikan semua konfigurasi sudah benar sebelum memulai training.</p>"
        "</div>"
    )
    
    # Buat container untuk informasi konfigurasi yang sudah diatur
    config_info = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            max_height='200px',
            overflow_y='auto'
        )
    )
    
    # Pastikan config tidak None
    config_data = config or {}
    
    # Tampilkan informasi konfigurasi yang sudah diatur
    with config_info:
        display(HTML(f"""
        <h4 style='margin-top:0;'>{ICONS.get('info', 'ℹ️')} Informasi Konfigurasi</h4>
        <p>Training akan menggunakan konfigurasi yang sudah diatur pada modul-modul sebelumnya:</p>
        <ul>
            <li><b>Backbone:</b> {config_data.get('model', {}).get('backbone', 'efficientnet_b4')}</li>
            <li><b>Batch Size:</b> {config_data.get('hyperparameters', {}).get('batch_size', 16)}</li>
            <li><b>Epochs:</b> {config_data.get('hyperparameters', {}).get('epochs', 100)}</li>
            <li><b>Learning Rate:</b> {config_data.get('hyperparameters', {}).get('learning_rate', 0.001)}</li>
        </ul>
        <p>Untuk mengubah konfigurasi, gunakan cell konfigurasi model, hyperparameter, dan training strategy.</p>
        """))
    
    # Komponen utama dengan layout yang lebih baik
    main_box = widgets.VBox([
        header,
        status_panel,
        config_info,
        create_divider(),
        action_buttons['container'],
        progress_components['progress_container'],
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('chart', '📊')} Metrik Training</h4>"),
        chart_output,
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('log', '📝')} Log Training</h4>"),
        log_components['log_accordion'],
        help_panel
    ], layout=widgets.Layout(width='100%', padding='15px'))
    
    # Kumpulkan semua komponen
    ui_components = {
        'main_container': main_box,
        'main_box': main_box,  # Untuk kompatibilitas
        'ui': main_box,  # Untuk kompatibilitas
        'header': header,
        'status_panel': status_panel,
        'info_box': info_box,
        'config_info': config_info,
        'metrics_box': metrics_box,
        'chart_output': chart_output,
        'create_metrics_chart': create_metrics_chart,
        'module_name': 'training',
        'logger': logger,
        
        # Simpan konfigurasi yang diambil dari modul-modul sebelumnya
        'config': config
    }
    
    # Tambahkan komponen action buttons
    ui_components.update({
        'start_button': action_buttons['primary_button'],
        'stop_button': action_buttons['stop_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        'button_container': action_buttons['container'],
        # Tambahkan tombol reset untuk reset form saja, bukan untuk save/reset konfigurasi
        'reset_form_button': action_buttons.get('reset_button')
    })
    
    # Tambahkan komponen progress tracking
    ui_components.update({
        'progress_bar': progress_components['progress_bar'],
        'progress_container': progress_components['progress_container'],
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label')
    })
    
    # Tambahkan komponen log
    ui_components.update({
        'status': log_components['log_output'],
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion']
    })
    
    return ui_components
