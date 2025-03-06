"""
File: smartcash/ui_components/data_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk data handling, termasuk informasi dataset dan utilitas split dataset.
"""

import ipywidgets as widgets
from IPython.display import HTML
from typing import Dict, Any

def create_dataset_info_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk menampilkan informasi dataset.
    
    Returns:
        Dictionary berisi komponen UI untuk bagian informasi dataset
    """
    # Tombol refresh informasi dataset
    refresh_info_button = widgets.Button(
        description='Refresh Info Dataset',
        button_style='info',
        icon='sync'
    )
    
    # Output area untuk menampilkan informasi
    info_output = widgets.Output()
    
    # Header dengan styling
    header = widgets.HTML("<h2>ℹ️ Informasi Dataset</h2>")
    description = widgets.HTML("<p>Pada bagian ini Anda dapat melihat statistik dan distribusi dataset.</p>")
    
    # Container untuk komponen UI
    ui_container = widgets.VBox([
        header,
        description,
        refresh_info_button,
        info_output
    ])
    
    # Return components dictionary
    return {
        'ui': ui_container,
        'refresh_info_button': refresh_info_button,
        'info_output': info_output
    }

def create_split_dataset_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk split dataset.
    
    Returns:
        Dictionary berisi komponen UI untuk bagian split dataset
    """
    # Widget untuk mengontrol split
    train_ratio_slider = widgets.FloatSlider(
        value=0.7,
        min=0.5,
        max=0.9,
        step=0.05,
        description='Ratio Train:',
        style={'description_width': 'initial'}
    )

    valid_ratio_slider = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.05,
        description='Ratio Valid:',
        style={'description_width': 'initial'}
    )

    test_ratio_slider = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.05,
        description='Ratio Test:',
        style={'description_width': 'initial'}
    )

    total_ratio_text = widgets.HTML(
        value="<b>Total Ratio: 1.0</b> ✅",
    )
    
    # Tombol untuk melakukan split dataset
    split_button = widgets.Button(
        description='Split Dataset',
        button_style='primary',
        icon='random'
    )
    
    # Output area
    split_status_output = widgets.Output()
    
    # Header dengan styling
    header = widgets.HTML("<h2>🔪 Split Dataset</h2>")
    description = widgets.HTML("<p>Bagi dataset menjadi beberapa bagian untuk training, validasi, dan testing.</p>")
    
    # Container untuk komponen UI
    ratio_sliders = widgets.VBox([
        train_ratio_slider,
        valid_ratio_slider,
        test_ratio_slider,
        total_ratio_text
    ])
    
    ui_container = widgets.VBox([
        header,
        description,
        ratio_sliders,
        split_button,
        split_status_output
    ])
    
    # Return components dictionary
    return {
        'ui': ui_container,
        'train_ratio_slider': train_ratio_slider,
        'valid_ratio_slider': valid_ratio_slider,
        'test_ratio_slider': test_ratio_slider,
        'total_ratio_text': total_ratio_text,
        'split_button': split_button,
        'split_status_output': split_status_output
    }

def create_data_utils_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk fungsi utilitas data.
    
    Returns:
        Dictionary berisi komponen UI untuk bagian utilitas data
    """
    # UI untuk visualisasi batch
    visualize_header = widgets.HTML("<h3>🖼️ Visualisasi Batch</h3>")
    
    # Dropdown untuk memilih split
    split_dropdown = widgets.Dropdown(
        options=[('Training', 'train'), ('Validation', 'valid'), ('Testing', 'test')],
        value='train',
        description='Dataset Split:',
        style={'description_width': 'initial'}
    )
    
    # Slider untuk jumlah gambar
    num_images_slider = widgets.IntSlider(
        value=4,
        min=1,
        max=16,
        step=1,
        description='Jumlah Gambar:',
        style={'description_width': 'initial'}
    )
    
    # Tombol visualisasi
    visualize_button = widgets.Button(
        description='Visualisasi Batch',
        button_style='info',
        icon='image'
    )
    
    # Output area
    visualization_output = widgets.Output()
    
    # Header dengan styling
    header = widgets.HTML("<h2>🛠️ Utilitas Data</h2>")
    description = widgets.HTML("<p>Fungsi-fungsi utilitas untuk membantu proses data handling.</p>")
    
    # Container untuk komponen UI
    visualize_controls = widgets.VBox([
        visualize_header,
        widgets.HBox([split_dropdown, num_images_slider]),
        visualize_button,
        visualization_output
    ])
    
    ui_container = widgets.VBox([
        header,
        description,
        visualize_controls
    ])
    
    # Return components dictionary
    return {
        'ui': ui_container,
        'split_dropdown': split_dropdown,
        'num_images_slider': num_images_slider,
        'visualize_button': visualize_button,
        'visualization_output': visualization_output
    }

def create_data_handling_ui() -> Dict[str, Any]:
    """
    Buat komponen UI lengkap untuk data handling, dengan tabs untuk berbagai fungsi.
    
    Returns:
        Dictionary berisi komponen UI untuk keseluruhan data handling
    """
    # Buat komponen UI untuk setiap bagian
    info_components = create_dataset_info_ui()
    split_components = create_split_dataset_ui()
    utils_components = create_data_utils_ui()
    
    # Buat tab untuk menampilkan berbagai fungsi
    tab = widgets.Tab()
    tab.children = [
        info_components['ui'],
        split_components['ui'],
        utils_components['ui']
    ]
    
    tab.set_title(0, "📊 Informasi Dataset")
    tab.set_title(1, "🔪 Split Dataset")
    tab.set_title(2, "🛠️ Utilitas Data")
    
    # Gabungkan semua komponen dalam struktur yang lengkap
    return {
        'ui': tab,
        'tab': tab,
        'info_components': info_components,
        'split_components': split_components,
        'utils_components': utils_components
    }