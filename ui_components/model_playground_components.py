"""
File: ui_components/model_playground_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk model playground, memungkinkan pengujian interaktif model.
"""

import ipywidgets as widgets
from IPython.display import display, HTML

def create_model_selector_controls():
    """
    Buat kontrol untuk memilih konfigurasi model.
    
    Returns:
        Dictionary berisi widget untuk memilih backbone dan mode deteksi
    """
    # Widget untuk memilih backbone
    backbone_selector = widgets.Dropdown(
        options=[
            ('EfficientNet-B4', 'efficientnet'),
            ('CSPDarknet (YOLOv5)', 'cspdarknet')
        ],
        value='efficientnet',
        description='Backbone:',
        style={'description_width': 'initial'}
    )
    
    # Widget untuk memilih mode deteksi
    detection_mode_selector = widgets.Dropdown(
        options=[
            ('Single Layer (Banknote)', 'single'),
            ('Multi Layer (+ Nominal & Security)', 'multi')
        ],
        value='single',
        description='Mode Deteksi:',
        style={'description_width': 'initial'}
    )
    
    return {
        'backbone_selector': backbone_selector,
        'detection_mode_selector': detection_mode_selector
    }

def create_model_option_controls():
    """
    Buat kontrol untuk opsi-opsi model tambahan.
    
    Returns:
        Dictionary berisi widget untuk konfigurasi model tambahan
    """
    # Widget untuk mengontrol opsi lain
    pretrained_checkbox = widgets.Checkbox(
        value=True,
        description='Gunakan Pretrained Weights',
        style={'description_width': 'initial'}
    )
    
    # Tambahan UI untuk pengaturan inferensi
    img_size_slider = widgets.IntSlider(
        value=640,
        min=320,
        max=1280,
        step=32,
        description='Ukuran Gambar:',
        style={'description_width': 'initial'}
    )
    
    return {
        'pretrained_checkbox': pretrained_checkbox,
        'img_size_slider': img_size_slider
    }

def create_model_test_controls():
    """
    Buat tombol untuk mencoba model.
    
    Returns:
        Dictionary berisi tombol test model
    """
    # Tombol untuk mencoba model
    test_model_button = widgets.Button(
        description='Buat & Test Model',
        button_style='success',
        icon='play'
    )
    
    return {
        'test_model_button': test_model_button
    }

def create_model_playground_ui():
    """
    Buat UI lengkap untuk model playground.
    
    Returns:
        Dictionary berisi semua komponen UI untuk model playground
    """
    # Buat header
    header = widgets.HTML("<h2>🧪 Model Testing Playground</h2>")
    description = widgets.HTML("<p>Coba berbagai konfigurasi model dan lihat performanya.</p>")
    
    # Buat komponen seleksi model
    selector_controls = create_model_selector_controls()
    
    # Buat komponen opsi model
    option_controls = create_model_option_controls()
    
    # Buat tombol test
    test_controls = create_model_test_controls()
    
    # Buat output area
    model_test_output = widgets.Output()
    
    # Susun layout UI
    left_column = widgets.VBox([
        selector_controls['backbone_selector'],
        selector_controls['detection_mode_selector'],
    ])
    
    right_column = widgets.VBox([
        option_controls['pretrained_checkbox'],
        option_controls['img_size_slider'],
    ])
    
    controls_layout = widgets.HBox([left_column, right_column])
    
    # Gabungkan semua komponen dalam layout utama
    main_ui = widgets.VBox([
        header,
        description,
        controls_layout,
        test_controls['test_model_button'],
        model_test_output
    ])
    
    # Return struktur UI dan komponen individual untuk handler
    return {
        'ui': main_ui,
        'output': model_test_output,
        'backbone_selector': selector_controls['backbone_selector'],
        'detection_mode_selector': selector_controls['detection_mode_selector'],
        'pretrained_checkbox': option_controls['pretrained_checkbox'],
        'img_size_slider': option_controls['img_size_slider'],
        'test_model_button': test_controls['test_model_button']
    }