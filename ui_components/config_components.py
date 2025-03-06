"""
smartcash/ui_components/config_components.py
Author: Alfrida Sabar

Komponen UI untuk konfigurasi dan pengaturan global SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, List, Any, Optional, Callable
import yaml
from pathlib import Path
import os
import torch
from datetime import datetime

from smartcash.utils.ui_utils import (
    create_header, 
    create_info_alert,
    create_section_title,
    create_status_indicator
)

def create_global_config_ui(config: Dict[str, Any], 
                          base_dir: Path,
                          active_layers: List[str],
                          available_layers: List[str]) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk konfigurasi global.
    
    Args:
        config: Dictionary konfigurasi saat ini
        base_dir: Path direktori dasar project
        active_layers: List layer yang aktif saat ini
        available_layers: List semua layer yang tersedia
        
    Returns:
        Dictionary berisi komponen UI
    """
    # Buat header
    header = create_header("🔧 Konfigurasi Global", 
                        "Atur konfigurasi global untuk SmartCash")
    
    # Informasi system
    system_info = []
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        gpu_info = create_status_indicator(
            'success', 
            f"GPU: {gpu_name}, Memory: {gpu_memory}"
        )
        system_info.append(gpu_info)
    else:
        gpu_info = create_status_indicator(
            'warning',
            "GPU tidak terdeteksi, menggunakan CPU"
        )
        system_info.append(gpu_info)
    
    # Direktori dasar
    dirs_info = widgets.HTML(
        f"""
        <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <p><b>📂 Direktori:</b></p>
            <ul>
                <li>Base: <code>{base_dir}</code></li>
                <li>Config: <code>{base_dir / 'configs'}</code></li>
                <li>Data: <code>{base_dir / 'data'}</code></li>
                <li>Output: <code>{base_dir / 'runs'}</code></li>
            </ul>
        </div>
        """
    )
    
    # Backbone selection
    backbone_dropdown = widgets.Dropdown(
        options=[
            ('EfficientNet-B4', 'efficientnet'),
            ('CSPDarknet (YOLOv5)', 'cspdarknet')
        ],
        value=config.get('model', {}).get('backbone', 'efficientnet'),
        description='Backbone:',
        style={'description_width': 'initial'}
    )
    
    # Pretrained checkbox
    pretrained_checkbox = widgets.Checkbox(
        value=config.get('model', {}).get('pretrained', True),
        description='Gunakan pretrained weights',
        style={'description_width': 'initial'}
    )
    
    # Image size slider
    img_size_slider = widgets.IntSlider(
        value=config.get('model', {}).get('img_size', [640, 640])[0],
        min=320,
        max=1280,
        step=32,
        description='Ukuran Gambar:',
        style={'description_width': 'initial'}
    )
    
    # Batch size slider
    batch_size_slider = widgets.IntSlider(
        value=config.get('model', {}).get('batch_size', 16),
        min=1,
        max=64,
        step=1,
        description='Batch Size:',
        style={'description_width': 'initial'}
    )
    
    # Workers slider
    workers_slider = widgets.IntSlider(
        value=config.get('model', {}).get('workers', 4),
        min=0,
        max=16,
        step=1,
        description='Workers:',
        style={'description_width': 'initial'}
    )
    
    # Learning rate dropdown
    lr_dropdown = widgets.Dropdown(
        options=[
            ('0.001 (default)', 1e-3),
            ('0.0001 (lebih stabil)', 1e-4),
            ('0.01 (lebih cepat)', 1e-2),
            ('0.00001 (fine-tuning)', 1e-5)
        ],
        value=config.get('training', {}).get('learning_rate', 1e-4),
        description='Learning Rate:',
        style={'description_width': 'initial'}
    )
    
    # Epochs slider
    epochs_slider = widgets.IntSlider(
        value=config.get('training', {}).get('epochs', 30),
        min=1,
        max=100,
        step=1,
        description='Epochs:',
        style={'description_width': 'initial'}
    )
    
    # Optimizer dropdown
    optimizer_dropdown = widgets.Dropdown(
        options=[
            ('AdamW (default)', 'adamw'),
            ('Adam', 'adam'),
            ('SGD', 'sgd')
        ],
        value=config.get('training', {}).get('optimizer', 'adamw'),
        description='Optimizer:',
        style={'description_width': 'initial'}
    )
    
    # Scheduler dropdown
    scheduler_dropdown = widgets.Dropdown(
        options=[
            ('Cosine Annealing', 'cosine'),
            ('Reduce On Plateau', 'plateau'),
            ('Step LR', 'step')
        ],
        value=config.get('training', {}).get('scheduler', 'cosine'),
        description='Scheduler:',
        style={'description_width': 'initial'}
    )
    
    # Layer selection
    layer_selection = widgets.SelectMultiple(
        options=available_layers,
        value=active_layers,
        description='Active Layers:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', height='100px')
    )
    
    # Layer info
    layer_info = widgets.HTML(
        f"""
        <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <p><b>🔍 Layer yang diaktifkan:</b> {', '.join(active_layers)}</p>
        </div>
        """
    )
    
    # Save config button
    save_config_button = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='primary',
        icon='save'
    )
    
    # Output area
    output_area = widgets.Output()
    
    # Organize UI layout
    model_section = widgets.VBox([
        create_section_title("Model", "🤖"),
        widgets.HBox([backbone_dropdown, pretrained_checkbox]),
        widgets.HBox([img_size_slider, batch_size_slider]),
        workers_slider
    ])
    
    training_section = widgets.VBox([
        create_section_title("Training", "🔄"),
        widgets.HBox([lr_dropdown, epochs_slider]),
        widgets.HBox([optimizer_dropdown, scheduler_dropdown])
    ])
    
    layer_section = widgets.VBox([
        create_section_title("Layers", "🔍"),
        layer_selection,
        layer_info
    ])
    
    button_section = widgets.VBox([
        save_config_button,
        output_area
    ])
    
    # Main UI container
    ui_container = widgets.VBox([
        header,
        widgets.VBox(system_info),
        dirs_info,
        model_section,
        training_section,
        layer_section,
        button_section
    ])
    
    # Return components dictionary
    return {
        'ui': ui_container,
        'backbone_dropdown': backbone_dropdown,
        'pretrained_checkbox': pretrained_checkbox,
        'img_size_slider': img_size_slider,
        'batch_size_slider': batch_size_slider,
        'workers_slider': workers_slider,
        'lr_dropdown': lr_dropdown,
        'epochs_slider': epochs_slider,
        'optimizer_dropdown': optimizer_dropdown,
        'scheduler_dropdown': scheduler_dropdown,
        'layer_selection': layer_selection,
        'layer_info': layer_info,
        'save_config_button': save_config_button,
        'output_area': output_area
    }

def create_pipeline_config_ui(config: Dict[str, Any], 
                             is_colab: bool = False,
                             api_key_exists: bool = False) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk konfigurasi pipeline.
    
    Args:
        config: Dictionary konfigurasi saat ini
        is_colab: Apakah berjalan di Google Colab
        api_key_exists: Apakah API key sudah ada
        
    Returns:
        Dictionary berisi komponen UI
    """
    # Buat header
    header = create_header("⚙️ Konfigurasi Pipeline", 
                        "Atur parameter utama untuk pipeline SmartCash")
    
    # API key status
    api_key_status = widgets.HTML(
        value=f"""
        <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: monospace;">
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: bold; margin-right: 10px;">Roboflow API Key:</span>
                <span style="background-color: {'#d4edda' if api_key_exists else '#f8d7da'}; 
                       color: {'#155724' if api_key_exists else '#721c24'}; 
                       padding: 2px 8px; border-radius: 3px;">
                    {'✅ Terdeteksi' if api_key_exists else '❌ Tidak ditemukan'}
                </span>
            </div>
            <div style="font-size: 0.9em; color: #6c757d;">
                <strong>Tips:</strong> Simpan API key di Google Colab Secret dengan nama <code>ROBOFLOW_API_KEY</code><br>
                <ol>
                    <li>Klik ikon 🔑 di sidebar sebelah kiri</li>
                    <li>Klik "+ Add new secret"</li>
                    <li>Masukkan <code>ROBOFLOW_API_KEY</code> sebagai nama</li>
                    <li>Masukkan API key dari akun Roboflow Anda</li>
                    <li>Klik "Save"</li>
                    <li>Restart runtime untuk menerapkan perubahan</li>
                </ol>
            </div>
        </div>
        """
    )
    
    # Backbone Selection
    backbone_dropdown = widgets.Dropdown(
        options=[('EfficientNet-B4', 'efficientnet'), ('CSPDarknet (Default YOLOv5)', 'cspdarknet')],
        value=config.get('model', {}).get('backbone', 'efficientnet'),
        description='Backbone:',
        style={'description_width': 'initial'}
    )

    # Batch Size Selection
    batch_size_slider = widgets.IntSlider(
        value=config.get('training', {}).get('batch_size', 16),
        min=4,
        max=64,
        step=4,
        description='Batch Size:',
        style={'description_width': 'initial'}
    )

    # Epochs Slider
    epochs_slider = widgets.IntSlider(
        value=config.get('training', {}).get('epochs', 30),
        min=5,
        max=100,
        step=5,
        description='Epochs:',
        style={'description_width': 'initial'}
    )

    # Learning Rate
    lr_dropdown = widgets.Dropdown(
        options=[('0.001 (Default)', 0.001), ('0.01', 0.01), ('0.0001', 0.0001)],
        value=config.get('training', {}).get('learning_rate', 0.001),
        description='Learning Rate:',
        style={'description_width': 'initial'}
    )

    # Data Source
    data_source_radio = widgets.RadioButtons(
        options=['roboflow', 'local'],
        value=config.get('data', {}).get('source', 'roboflow'),
        description='Sumber Data:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    )

    # Detection Mode
    detection_mode_radio = widgets.RadioButtons(
        options=['single', 'multi'],
        value=config.get('detection_mode', 'single'),
        description='Mode Deteksi:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    )

    # Roboflow Project Details
    workspace_input = widgets.Text(
        value=config.get('data', {}).get('roboflow', {}).get('workspace', ""), 
        description='Roboflow Workspace:',
        layout=widgets.Layout(width='500px'),
        style={'description_width': 'initial'}
    )

    project_input = widgets.Text(
        value=config.get('data', {}).get('roboflow', {}).get('project', ""), 
        description='Roboflow Project:',
        layout=widgets.Layout(width='500px'),
        style={'description_width': 'initial'}
    )

    # Version sebagai IntSlider
    version_input = widgets.IntSlider(
        value=config.get('data', {}).get('roboflow', {}).get('version', 3),
        min=1,
        max=100,
        step=1,
        description='Version:',
        style={'description_width': 'initial'}
    )
    
    # Save button
    save_config_button = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='primary',
        icon='save'
    )

    # Reload button
    reload_config_button = widgets.Button(
        description='Muat Ulang Konfigurasi',
        button_style='warning',
        icon='sync'
    )
    
    # Output area
    output_area = widgets.Output()
    
    # Organize UI layout
    params_box = widgets.VBox([
        api_key_status,
        widgets.HBox([backbone_dropdown, data_source_radio]),
        widgets.HBox([batch_size_slider, epochs_slider]),
        widgets.HBox([lr_dropdown, detection_mode_radio]),
        workspace_input,
        project_input,
        version_input
    ])
    
    buttons_box = widgets.HBox([save_config_button, reload_config_button])
    
    # Main UI container
    ui_container = widgets.VBox([
        header,
        params_box,
        buttons_box,
        output_area
    ])
    
    # Return components dictionary
    return {
        'ui': ui_container,
        'backbone_dropdown': backbone_dropdown,
        'batch_size_slider': batch_size_slider,
        'epochs_slider': epochs_slider,
        'lr_dropdown': lr_dropdown,
        'data_source_radio': data_source_radio,
        'detection_mode_radio': detection_mode_radio,
        'workspace_input': workspace_input,
        'project_input': project_input,
        'version_input': version_input,
        'save_config_button': save_config_button,
        'reload_config_button': reload_config_button,
        'output_area': output_area
    }