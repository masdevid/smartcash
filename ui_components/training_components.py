"""
File: smartcash/ui_components/training_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk visualisasi dan kontrol proses training model, serta konfigurasi pipeline training.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from datetime import datetime, timedelta
import torch

# ===== COMPONENTS FOR CELL 93 (TRAINING EXECUTION) =====

def create_training_controls():
    """
    Buat kontrol untuk menjalankan dan menghentikan proses training.
    
    Returns:
        Dictionary berisi widget tombol kontrol training
    """
    # Tombol start training
    start_training_button = widgets.Button(
        description='Mulai Training',
        button_style='success',
        icon='play',
        tooltip='Mulai proses training model'
    )
    
    # Tombol stop training
    stop_training_button = widgets.Button(
        description='Hentikan Training',
        button_style='danger',
        icon='stop',
        disabled=True,
        tooltip='Hentikan proses training'
    )
    
    # Checkbox untuk resume training
    resume_checkbox = widgets.Checkbox(
        value=False,
        description='Resume dari checkpoint terbaik',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Batch size override
    batch_size_dropdown = widgets.Dropdown(
        options=[8, 16, 32, 64],
        value=16,  # Default value, akan diupdate dari config
        description='Batch Size:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='30%')
    )
    
    return {
        'start_training_button': start_training_button,
        'stop_training_button': stop_training_button,
        'resume_checkbox': resume_checkbox,
        'batch_size_dropdown': batch_size_dropdown
    }

def create_drive_backup_control(is_colab=False):
    """
    Buat kontrol untuk backup checkpoint ke Google Drive.
    
    Args:
        is_colab: Boolean yang menunjukkan apakah berjalan di Colab
        
    Returns:
        Widget checkbox untuk backup ke Drive
    """
    import os
    
    # Opsi Google Drive backup
    drive_backup_checkbox = widgets.Checkbox(
        value=True if is_colab and os.path.exists("/content/drive") else False,
        description='Backup checkpoint ke Google Drive',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
        disabled=not (is_colab and os.path.exists("/content/drive"))
    )
    
    return drive_backup_checkbox

def create_status_display():
    """
    Buat widgets untuk menampilkan status training.
    
    Returns:
        Dictionary berisi widget untuk menampilkan status
    """
    # Status text
    status_text = widgets.HTML(
        value="<p><b>Status:</b> Idle</p><p><b>Epoch:</b> 0/?</p><p><b>Best Val Loss:</b> -</p>"
    )
    
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        step=1,
        description='Progress:',
        bar_style='info',
        orientation='horizontal'
    )
    
    return {
        'status_text': status_text,
        'progress_bar': progress_bar
    }

def create_visualization_tabs():
    """
    Buat tabs untuk visualisasi training.
    
    Returns:
        Dictionary berisi tabs dan output areas
    """
    # Output areas for visualizations
    live_plot_tab = widgets.Output()
    metrics_tab = widgets.Output()
    status_tab = widgets.Output()
    
    # Create tabs
    tabs = widgets.Tab([live_plot_tab, metrics_tab, status_tab])
    tabs.set_title(0, 'Live Plot')
    tabs.set_title(1, 'Metrics')
    tabs.set_title(2, 'Status')
    
    return {
        'tabs': tabs,
        'live_plot_tab': live_plot_tab,
        'metrics_tab': metrics_tab,
        'status_tab': status_tab
    }

def create_training_ui(config=None, is_colab=False):
    """
    Buat UI lengkap untuk training model.
    
    Args:
        config: Dictionary konfigurasi (untuk mengambil nilai default)
        is_colab: Boolean yang menunjukkan apakah berjalan di Colab
        
    Returns:
        Dictionary berisi komponen UI untuk training
    """
    # Set batch size default dari config jika tersedia
    default_batch_size = 16
    if config and 'training' in config and 'batch_size' in config['training']:
        default_batch_size = config['training']['batch_size']
    
    # Buat header dengan styling
    header = widgets.HTML("<h2>🚀 Training Model</h2>")
    description = widgets.HTML("<p>Jalankan proses training model dengan konfigurasi yang telah ditetapkan.</p>")
    
    # Buat kontrol training
    controls = create_training_controls()
    # Update batch size default
    controls['batch_size_dropdown'].value = default_batch_size
    
    # Buat kontrol Google Drive backup
    drive_backup_checkbox = create_drive_backup_control(is_colab)
    
    # Buat status display
    status_display = create_status_display()
    
    # Buat output untuk training
    training_output = widgets.Output()
    
    # Buat tabs visualisasi
    viz_tabs = create_visualization_tabs()
    
    # Susun layout UI
    controls_layout = widgets.VBox([
        widgets.HTML("<h3>⚙️ Kontrol Training</h3>"),
        widgets.HBox([controls['resume_checkbox'], drive_backup_checkbox]),
        widgets.HBox([controls['batch_size_dropdown']]),
        widgets.HBox([controls['start_training_button'], controls['stop_training_button']]),
        status_display['progress_bar'],
        status_display['status_text']
    ])
    
    viz_layout = widgets.VBox([
        widgets.HTML("<h3>📊 Visualisasi Training</h3>"),
        viz_tabs['tabs']
    ])
    
    # Gabungkan semua komponen dalam layout utama
    main_ui = widgets.VBox([
        header,
        description,
        controls_layout,
        training_output,
        viz_layout
    ])
    
    # Return struktur UI dan komponen individual untuk handler
    return {
        'ui': main_ui,
        'output': training_output,
        'start_button': controls['start_training_button'],
        'stop_button': controls['stop_training_button'],
        'resume_checkbox': controls['resume_checkbox'],
        'batch_size_dropdown': controls['batch_size_dropdown'],
        'drive_backup_checkbox': drive_backup_checkbox,
        'status_text': status_display['status_text'],
        'progress_bar': status_display['progress_bar'],
        'tabs': viz_tabs['tabs'],
        'live_plot_tab': viz_tabs['live_plot_tab'],
        'metrics_tab': viz_tabs['metrics_tab'],
        'status_tab': viz_tabs['status_tab']
    }

# ===== COMPONENTS FOR CELL 91 (TRAINING PIPELINE) =====

def create_training_pipeline_ui():
    """
    Buat komponen UI untuk pipeline training.
    
    Returns:
        Dictionary berisi komponen UI untuk pipeline training
    """
    # Buat header dengan styling
    header = widgets.HTML("<h2>🚀 Pipeline Training</h2>")
    description = widgets.HTML("<p>Konfigurasi dan inisialisasi pipeline training.</p>")
    
    # Status training dengan gaya yang lebih baik
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        step=1,
        description='Progress:',
        bar_style='info',
        orientation='horizontal'
    )
    
    # Info text dengan format HTML yang lebih baik
    info_text = widgets.HTML(
        value="""
        <div style="padding: 10px; border-radius: 5px; background-color: #f5f5f5; margin-bottom: 10px">
            <p><b>Status:</b> <span style="color: gray">Idle</span></p>
            <p><b>Epoch:</b> 0/?</p>
            <p><b>Best Val Loss:</b> -</p>
            <p><b>Device:</b> {}</p>
        </div>
        """.format("GPU" if torch.cuda.is_available() else "CPU")
    )
    
    # Button untuk cek status dengan ikon
    check_status_button = widgets.Button(
        description='Refresh Status',
        button_style='info',
        icon='sync',
        tooltip='Periksa status training terbaru'
    )
    
    # Output untuk status training
    status_output = widgets.Output()
    
    # Gabungkan elemen UI
    status_ui = widgets.VBox([
        widgets.HTML('<h3 style="margin-bottom: 10px;">🔄 Status Training</h3>'),
        progress_bar,
        info_text,
        check_status_button,
        status_output
    ])
    
    # Return components dictionary
    return {
        'ui': status_ui,
        'progress_bar': progress_bar,
        'info_text': info_text,
        'check_status_button': check_status_button,
        'status_output': status_output
    }

# ===== COMPONENTS FOR CELL 92 (TRAINING CONFIGURATION) =====

def create_training_config_ui(config=None):
    """
    Buat komponen UI untuk konfigurasi training.
    
    Args:
        config: Dictionary konfigurasi default (optional)
        
    Returns:
        Dictionary berisi komponen UI untuk konfigurasi training
    """
    if config is None:
        config = {}
    
    # Header dengan styling
    header = widgets.HTML(
        value='<h2 style="color: #3498db; margin-bottom: 15px;">⚙️ Konfigurasi Training</h2>' +
              '<p style="color: #555; margin-bottom: 20px;">Sesuaikan parameter training model sebelum mulai proses training.</p>'
    )

    # Hyperparameter epochs dengan tooltip informatif
    epochs_slider = widgets.IntSlider(
        value=config.get('training', {}).get('epochs', 30),
        min=1,
        max=100,
        step=1,
        description='Epochs:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
        tooltip='Jumlah iterasi training pada seluruh dataset'
    )

    # Batch size dengan slider dinamis berdasarkan ketersediaan GPU
    max_batch = 32 if torch.cuda.is_available() else 16
    batch_size_slider = widgets.IntSlider(
        value=min(config.get('training', {}).get('batch_size', 16), max_batch),
        min=1,
        max=max_batch,
        step=1,
        description='Batch Size:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
        tooltip='Jumlah sampel yang diproses dalam satu iterasi'
    )

    # Learning rate dengan lebih banyak pilihan
    lr_dropdown = widgets.Dropdown(
        options=[
            ('0.001 (default)', 0.001),
            ('0.01 (lebih cepat)', 0.01),
            ('0.0001 (lebih stabil)', 0.0001),
            ('0.00001 (fine-tuning)', 0.00001),
            ('0.002 (warmup)', 0.002)
        ],
        value=config.get('training', {}).get('learning_rate', 0.001),
        description='Learning Rate:',
        style={'description_width': 'initial'},
        tooltip='Kecepatan pembelajaran model: nilai kecil=pembelajaran lambat tapi stabil'
    )

    # Optimizer dengan opsi tambahan
    optimizer_dropdown = widgets.Dropdown(
        options=[
            ('Adam (default)', 'adam'),
            ('AdamW (lebih robust)', 'adamw'),
            ('SGD (klasik)', 'sgd'),
            ('RMSprop (adaptive)', 'rmsprop')
        ],
        value=config.get('training', {}).get('optimizer', 'adam'),
        description='Optimizer:',
        style={'description_width': 'initial'},
        tooltip='Algoritma optimasi yang digunakan untuk memperbarui bobot model'
    )

    # Scheduler dengan opsi tambahan
    scheduler_dropdown = widgets.Dropdown(
        options=[
            ('ReduceLROnPlateau (default)', 'plateau'),
            ('StepLR (interval tetap)', 'step'),
            ('CosineAnnealing (smooth)', 'cosine'),
            ('OneCycleLR (warmup+annealing)', 'onecycle')
        ],
        value=config.get('training', {}).get('scheduler', 'plateau'),
        description='Scheduler:',
        style={'description_width': 'initial'},
        tooltip='Pengaturan perubahan learning rate selama training'
    )

    # Early stopping patience dengan keterangan
    early_stopping_slider = widgets.IntSlider(
        value=config.get('training', {}).get('early_stopping_patience', 5),
        min=1,
        max=20,
        step=1,
        description='Early Stopping:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%'),
        tooltip='Jumlah epoch tanpa peningkatan sebelum training dihentikan'
    )

    # Weight decay untuk regularisasi
    weight_decay_dropdown = widgets.Dropdown(
        options=[
            ('0.0005 (default)', 0.0005),
            ('0.001 (lebih kuat)', 0.001),
            ('0.0001 (lebih lemah)', 0.0001),
            ('0 (tanpa regularisasi)', 0)
        ],
        value=config.get('training', {}).get('weight_decay', 0.0005),
        description='Weight Decay:',
        style={'description_width': 'initial'},
        tooltip='Parameter regularisasi untuk mencegah overfitting'
    )

    # Strategi checkpoint
    save_every_slider = widgets.IntSlider(
        value=config.get('training', {}).get('save_every', 5),
        min=1,
        max=10,
        step=1,
        description='Save Every:',
        style={'description_width': 'initial'},
        tooltip='Simpan checkpoint setiap X epoch',
        layout=widgets.Layout(width='50%')
    )

    # Layer selection (multiselect)
    available_layers = ['banknote', 'nominal', 'security']
    active_layers = config.get('layers', ['banknote'])

    layer_selection = widgets.SelectMultiple(
        options=available_layers,
        value=active_layers,
        description='Layers Aktif:',
        style={'description_width': 'initial'},
        tooltip='Layer deteksi yang akan diaktifkan',
        layout=widgets.Layout(width='50%', height='80px')
    )

    # Experiment name input dengan generator otomatis
    default_name = f"{config.get('model', {}).get('backbone', 'efficientnet')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment_name_input = widgets.Text(
        value=default_name,
        placeholder='Nama eksperimen training',
        description='Nama Eksperimen:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%'),
        tooltip='Nama unik untuk mengidentifikasi eksperimen ini'
    )

    # Generate random name button
    generate_name_button = widgets.Button(
        description='Generate Nama',
        button_style='info',
        icon='random',
        tooltip='Generate nama eksperimen acak'
    )

    # Save config button
    save_config_button = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='primary',
        icon='save'
    )

    # Show learning rate schedule button
    show_lr_schedule_button = widgets.Button(
        description='Visualisasi LR Schedule',
        button_style='info',
        icon='line-chart',
        tooltip='Tampilkan visualisasi learning rate schedule untuk konfigurasi saat ini'
    )

    # Output areas
    config_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            max_height='200px',
            overflow='auto'
        )
    )
    
    lr_schedule_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0'
        )
    )

    # Group widget untuk parameter backbone
    backbone_dropdown = widgets.Dropdown(
        options=[
            ('EfficientNet-B4 (default)', 'efficientnet'),
            ('CSPDarknet (YOLOv5)', 'cspdarknet')
        ],
        value=config.get('model', {}).get('backbone', 'efficientnet'),
        description='Backbone:',
        style={'description_width': 'initial'},
        tooltip='Model backbone untuk ekstraksi fitur',
        layout=widgets.Layout(width='50%')
    )

    pretrained_checkbox = widgets.Checkbox(
        value=config.get('model', {}).get('pretrained', True),
        description='Gunakan pretrained weights',
        style={'description_width': 'initial'},
        tooltip='Menggunakan pretrained weights dari ImageNet untuk transfer learning'
    )
    
    backbone_group = widgets.VBox([
        widgets.HTML("<h3 style='color: #2980b9; margin-top: 15px;'>🔌 Konfigurasi Backbone</h3>"),
        widgets.HBox([backbone_dropdown, pretrained_checkbox])
    ])

    # Group widget untuk parameter training
    training_params_group = widgets.VBox([
        widgets.HTML("<h3 style='color: #2980b9;'>🔄 Parameter Training</h3>"),
        widgets.HBox([epochs_slider, batch_size_slider]),
        widgets.HBox([lr_dropdown, optimizer_dropdown]),
        widgets.HBox([scheduler_dropdown, weight_decay_dropdown]),
        widgets.HBox([early_stopping_slider, save_every_slider]),
        widgets.VBox([
            widgets.HTML("<h4 style='margin-top: 10px;'>🔍 Layers Deteksi</h4>"),
            layer_selection
        ])
    ])

    # Group widget untuk eksperimen
    experiment_group = widgets.VBox([
        widgets.HTML("<h3 style='color: #2980b9; margin-top: 15px;'>🧪 Konfigurasi Eksperimen</h3>"),
        widgets.HBox([experiment_name_input, generate_name_button]),
        widgets.HBox([save_config_button, show_lr_schedule_button])
    ])

    # Return components dictionary
    return {
        'ui': widgets.VBox([
            header,
            backbone_group,
            training_params_group,
            experiment_group,
            config_output,
            lr_schedule_output
        ]),
        'epochs_slider': epochs_slider,
        'batch_size_slider': batch_size_slider,
        'lr_dropdown': lr_dropdown,
        'optimizer_dropdown': optimizer_dropdown,
        'scheduler_dropdown': scheduler_dropdown,
        'early_stopping_slider': early_stopping_slider,
        'weight_decay_dropdown': weight_decay_dropdown,
        'save_every_slider': save_every_slider,
        'layer_selection': layer_selection,
        'experiment_name_input': experiment_name_input,
        'generate_name_button': generate_name_button,
        'save_config_button': save_config_button,
        'show_lr_schedule_button': show_lr_schedule_button,
        'config_output': config_output,
        'lr_schedule_output': lr_schedule_output,
        'backbone_dropdown': backbone_dropdown,
        'pretrained_checkbox': pretrained_checkbox
    }