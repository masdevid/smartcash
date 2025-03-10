"""
File: smartcash/ui_components/model_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk manajemen model SmartCash, termasuk inisialisasi, visualisasi, dan checkpoint management.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, Optional, List

def create_model_initialization_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk inisialisasi model.
    
    Returns:
        Dictionary berisi komponen UI untuk inisialisasi model
    """
    # Header dan deskripsi
    header = widgets.HTML("<h2>🧠 Inisialisasi Model</h2>")
    description = widgets.HTML("<p>Inisialisasi model dan konfigurasi parameter dasar.</p>")
    
    # Dropdown untuk memilih backbone
    backbone_dropdown = widgets.Dropdown(
        options=[
            ('EfficientNet-B4', 'efficientnet'),
            ('CSPDarknet (YOLOv5)', 'cspdarknet')
        ],
        value='efficientnet',
        description='Backbone:',
        style={'description_width': 'initial'}
    )
    
    # Dropdown untuk memilih mode deteksi
    detection_mode_dropdown = widgets.Dropdown(
        options=[
            ('Single Layer', 'single'),
            ('Multiple Layers', 'multi')
        ],
        value='single',
        description='Mode:',
        style={'description_width': 'initial'}
    )
    
    # Checkbox untuk pretrained weights
    pretrained_checkbox = widgets.Checkbox(
        value=True,
        description='Gunakan pretrained weights',
        style={'description_width': 'initial'}
    )
    
    # Slider untuk ukuran gambar
    img_size_slider = widgets.IntSlider(
        value=640,
        min=320,
        max=1280,
        step=32,
        description='Ukuran Gambar:',
        style={'description_width': 'initial'}
    )
    
    # Tombol inisialisasi
    initialize_button = widgets.Button(
        description='Inisialisasi Model',
        button_style='primary',
        icon='rocket'
    )
    
    # Output area
    output_area = widgets.Output()
    
    # Susun layout UI
    controls = widgets.VBox([
        widgets.HBox([backbone_dropdown, detection_mode_dropdown]),
        widgets.HBox([pretrained_checkbox, img_size_slider]),
        initialize_button
    ])
    
    # Gabungkan semua komponen
    ui_container = widgets.VBox([
        header,
        description,
        controls,
        output_area
    ])
    
    # Return komponen untuk digunakan di handler
    return {
        'ui': ui_container,
        'backbone_dropdown': backbone_dropdown,
        'detection_mode_dropdown': detection_mode_dropdown,
        'pretrained_checkbox': pretrained_checkbox,
        'img_size_slider': img_size_slider,
        'initialize_button': initialize_button,
        'output_area': output_area
    }

def create_model_visualizer_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk visualisasi model.
    
    Returns:
        Dictionary berisi komponen UI untuk visualisasi model
    """
    # Header dan deskripsi
    header = widgets.HTML("<h2>🔍 Visualisasi Model</h2>")
    description = widgets.HTML("<p>Visualisasi struktur dan parameter model berdasarkan konfigurasi.</p>")
    
    # Backbone selection
    backbone_select = widgets.Dropdown(
        options=[
            ('EfficientNet-B4', 'efficientnet'),
            ('CSPDarknet (YOLOv5)', 'cspdarknet')
        ],
        value='efficientnet',
        description='Backbone:',
        style={'description_width': 'initial'}
    )
    
    # Mode selection
    mode_select = widgets.Dropdown(
        options=[
            ('Single Layer', 'single'),
            ('Multiple Layers', 'multi')
        ],
        value='single',
        description='Mode:',
        style={'description_width': 'initial'}
    )
    
    # Visualization module selection
    viz_module_select = widgets.RadioButtons(
        options=[
            ('Full Model', 'full'),
            ('Backbone Only', 'backbone'),
            ('Parameters', 'parameters')
        ],
        value='full',
        description='Visualisasi:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    )
    
    # Create model button
    create_model_button = widgets.Button(
        description='Buat Model & Visualisasikan',
        button_style='primary',
        icon='rocket'
    )
    
    # Output area
    visualization_output = widgets.Output()
    
    # Arrange UI layout
    controls = widgets.VBox([
        widgets.HBox([backbone_select, mode_select]),
        viz_module_select,
        create_model_button
    ])
    
    # Combine all components
    ui_container = widgets.VBox([
        header,
        description,
        controls,
        visualization_output
    ])
    
    # Return components for use in handler
    return {
        'ui': ui_container,
        'backbone_select': backbone_select,
        'mode_select': mode_select,
        'viz_module_select': viz_module_select,
        'create_model_button': create_model_button,
        'visualization_output': visualization_output
    }

def create_checkpoint_manager_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk manajemen checkpoint model.
    
    Returns:
        Dictionary berisi komponen UI untuk manajemen checkpoint
    """
    # Header dan deskripsi
    header = widgets.HTML("<h2>📦 Model Checkpoints</h2>")
    description = widgets.HTML("<p>Kelola dan lihat checkpoint model yang tersimpan.</p>")
    
    # Tombol untuk melihat checkpoint
    list_checkpoints_button = widgets.Button(
        description='Lihat Checkpoints',
        button_style='info',
        icon='list'
    )
    
    # Tombol untuk membersihkan checkpoint
    cleanup_checkpoints_button = widgets.Button(
        description='Bersihkan Checkpoints',
        button_style='warning',
        icon='trash'
    )
    
    # Tombol untuk membandingkan checkpoint
    compare_button = widgets.Button(
        description='Bandingkan Checkpoints',
        button_style='primary',
        icon='exchange'
    )
    
    # Mount Google Drive button jika di Colab
    mount_drive_button = widgets.Button(
        description='Mount Google Drive',
        button_style='success',
        icon='folder'
    )
    
    # Output area
    checkpoints_output = widgets.Output()
    
    # Susun layout UI
    buttons = widgets.HBox([
        list_checkpoints_button, 
        cleanup_checkpoints_button, 
        compare_button
    ])
    
    # Gabungkan semua komponen
    ui_container = widgets.VBox([
        header,
        description,
        mount_drive_button,
        buttons,
        checkpoints_output
    ])
    
    # Return komponen untuk digunakan di handler
    return {
        'ui': ui_container,
        'list_checkpoints_button': list_checkpoints_button,
        'cleanup_checkpoints_button': cleanup_checkpoints_button,
        'compare_button': compare_button,
        'mount_drive_button': mount_drive_button,
        'checkpoints_output': checkpoints_output
    }

def create_model_optimization_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk optimasi model dan memori.
    
    Returns:
        Dictionary berisi komponen UI untuk optimasi model
    """
    # Header dan deskripsi
    header = widgets.HTML("<h2>🚀 Optimasi Model</h2>")
    description = widgets.HTML("<p>Tools untuk mengoptimalkan penggunaan memori dan performa model.</p>")
    
    # Tombol untuk memeriksa status memori
    check_memory_button = widgets.Button(
        description='Cek Status Memori',
        button_style='info',
        icon='server'
    )
    
    # Tombol untuk membersihkan memori GPU
    clear_memory_button = widgets.Button(
        description='Bersihkan Memori GPU',
        button_style='warning',
        icon='trash'
    )
    
    # Tombol untuk optimasi batch size
    optimize_button = widgets.Button(
        description='Optimasi Batch Size',
        button_style='success',
        icon='cogs'
    )
    
    # Tombol untuk manajemen cache
    clear_cache_button = widgets.Button(
        description='Bersihkan Cache',
        button_style='danger',
        icon='trash-alt'
    )
    
    verify_cache_button = widgets.Button(
        description='Verifikasi Cache',
        button_style='info',
        icon='check-circle'
    )
    
    # Output area
    memory_output = widgets.Output()
    
    # Susun layout UI
    memory_buttons = widgets.HBox([
        check_memory_button,
        clear_memory_button
    ])
    
    optimization_buttons = widgets.HBox([
        optimize_button
    ])
    
    cache_buttons = widgets.HBox([
        clear_cache_button,
        verify_cache_button
    ])
    
    # Gabungkan semua komponen
    ui_container = widgets.VBox([
        header,
        description,
        memory_buttons,
        optimization_buttons,
        cache_buttons,
        memory_output
    ])
    
    # Return komponen untuk digunakan di handler
    return {
        'ui': ui_container,
        'check_memory_button': check_memory_button,
        'clear_memory_button': clear_memory_button,
        'optimize_button': optimize_button,
        'clear_cache_button': clear_cache_button,
        'verify_cache_button': verify_cache_button,
        'memory_output': memory_output
    }

def create_model_exporter_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk ekspor model ke format produksi.
    
    Returns:
        Dictionary berisi komponen UI untuk ekspor model
    """
    # Header dan deskripsi
    header = widgets.HTML("<h2>📦 Ekspor Model</h2>")
    description = widgets.HTML("<p>Ekspor model checkpoint terbaik ke format yang dapat digunakan di produksi.</p>")
    
    # Selector untuk format ekspor
    export_format_selector = widgets.Dropdown(
        options=[
            ('TorchScript (PyTorch)', 'torchscript'),
            ('ONNX (Open Format)', 'onnx')
        ],
        value='torchscript',
        description='Format Ekspor:',
        style={'description_width': 'initial'}
    )
    
    # Opsi optimasi
    optimize_checkbox = widgets.Checkbox(
        value=True,
        description='Optimalkan untuk Inferensi',
        style={'description_width': 'initial'}
    )
    
    # Opsi ONNX
    onnx_opset_selector = widgets.IntSlider(
        value=12,
        min=10,
        max=14,
        step=1,
        description='ONNX Opset Version:',
        disabled=True,
        style={'description_width': 'initial'}
    )
    
    # Opsi salin ke Drive
    copy_to_drive_checkbox = widgets.Checkbox(
        value=True,
        description='Salin ke Google Drive',
        style={'description_width': 'initial'}
    )
    
    # Tombol ekspor
    export_button = widgets.Button(
        description='Ekspor Model',
        button_style='primary',
        icon='download'
    )
    
    # Output area
    export_output = widgets.Output()
    
    # Arrange UI layout
    options = widgets.VBox([
        export_format_selector,
        widgets.HBox([optimize_checkbox, onnx_opset_selector]),
        copy_to_drive_checkbox
    ])
    
    # Combine all components
    ui_container = widgets.VBox([
        header,
        description,
        options,
        export_button,
        export_output
    ])
    
    # Return components for use in handler
    return {
        'ui': ui_container,
        'export_format_selector': export_format_selector,
        'optimize_checkbox': optimize_checkbox,
        'onnx_opset_selector': onnx_opset_selector,
        'copy_to_drive_checkbox': copy_to_drive_checkbox,
        'export_button': export_button,
        'export_output': export_output
    }

def create_model_manager_ui() -> Dict[str, Any]:
    """
    Buat komponen UI lengkap untuk manajemen model dengan tab untuk berbagai fungsi.
    
    Returns:
        Dictionary berisi komponen UI untuk keseluruhan manajemen model
    """
    # Buat komponen UI untuk setiap bagian
    init_components = create_model_initialization_ui()
    visualizer_components = create_model_visualizer_ui()
    checkpoint_components = create_checkpoint_manager_ui()
    optimization_components = create_model_optimization_ui()
    exporter_components = create_model_exporter_ui()
    
    # Buat tab untuk menampilkan berbagai fungsi
    tab = widgets.Tab()
    tab.children = [
        init_components['ui'],
        visualizer_components['ui'],
        checkpoint_components['ui'],
        optimization_components['ui'],
        exporter_components['ui']
    ]
    
    tab.set_title(0, "🧠 Inisialisasi")
    tab.set_title(1, "🔍 Visualisasi")
    tab.set_title(2, "📦 Checkpoints")
    tab.set_title(3, "🚀 Optimasi")
    tab.set_title(4, "📦 Ekspor")
    
    # Gabungkan semua komponen dalam struktur yang lengkap
    return {
        'ui': tab,
        'tab': tab,
        'init_components': init_components,
        'visualizer_components': visualizer_components,
        'checkpoint_components': checkpoint_components,
        'optimization_components': optimization_components,
        'exporter_components': exporter_components
    }