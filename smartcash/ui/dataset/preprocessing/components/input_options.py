"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Description: Essential preprocessing input forms with modern UI structure
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.dataset.preprocessing.constants import (
    YOLO_PRESETS, DEFAULT_SPLITS, SUPPORTED_SPLITS, CleanupTarget
)

def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Create essential preprocessing forms with modern UI structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VBox container with input options
    """
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    validation_config = preprocessing_config.get('validation', {})
    data_config = config.get('data', {})
    
    # === YOLO PRESET SECTION ===
    
    # YOLO Preset dropdown (replaces resolution)
    current_preset = normalization_config.get('preset', 'yolov5s')
    preset_options = [(f"{preset} ({YOLO_PRESETS[preset]['target_size'][0]}x{YOLO_PRESETS[preset]['target_size'][1]})", preset) 
                     for preset in YOLO_PRESETS.keys()]
    
    resolution_dropdown = widgets.Dropdown(
        options=preset_options,
        value=current_preset if current_preset in YOLO_PRESETS else 'yolov5s',
        description='Preset YOLO:',
        style={'description_width': '90px'},
        tooltip='Pilih preset model YOLO untuk normalisasi gambar'
    )
    resolution_dropdown.layout = widgets.Layout(width='100%', margin='2px 0')
    
    # Normalization method
    method = normalization_config.get('method', 'minmax')
    normalization_dropdown = widgets.Dropdown(
        options=[
            ('Min-Max (0-1)', 'minmax'), 
            ('Z-Score', 'zscore'), 
            ('Robust', 'robust')
        ],
        value=method if method in ['minmax', 'zscore', 'robust'] else 'minmax',
        description='Metode:',
        style={'description_width': '90px'},
        tooltip='Pilih metode normalisasi untuk data gambar'
    )
    normalization_dropdown.layout = widgets.Layout(width='100%', margin='2px 0')
    
    # Preserve aspect ratio
    preserve_aspect_checkbox = widgets.Checkbox(
        value=normalization_config.get('preserve_aspect_ratio', True),
        description='Pertahankan Rasio Aspek (YOLO)',
        style={'description_width': 'initial'},
        tooltip='Pertahankan rasio aspek gambar saat melakukan resize'
    )
    preserve_aspect_checkbox.layout = widgets.Layout(width='100%', margin='2px 0')
    
    normalization_section = widgets.VBox([
        widgets.HTML(value="<div style='font-weight:bold;color:#2196F3;margin-bottom:6px;'>🎨 Normalisasi YOLO</div>"),
        resolution_dropdown,
        normalization_dropdown, 
        preserve_aspect_checkbox
    ])
    normalization_section.layout = widgets.Layout(width='48%', padding='8px')
    
    # === PROCESSING SECTION ===
    
    # Target splits
    target_splits = preprocessing_config.get('target_splits', DEFAULT_SPLITS)
    split_options = [(split.title(), split) for split in SUPPORTED_SPLITS]
    
    target_splits_select = widgets.SelectMultiple(
        options=split_options,
        value=tuple(target_splits) if isinstance(target_splits, list) else tuple(DEFAULT_SPLITS),
        description='Pembagian Data:',
        style={'description_width': '90px'},
        tooltip='Pilih pembagian data untuk preprocessing (train/val/test)'
    )
    target_splits_select.layout = widgets.Layout(width='100%', height='80px', margin='2px 0')
    
    # Batch size
    batch_size_input = widgets.BoundedIntText(
        value=preprocessing_config.get('batch_size', 32),
        min=1, max=256, step=1,
        description='Ukuran Batch:',
        style={'description_width': '90px'},
        tooltip='Jumlah sampel yang diproses dalam satu batch (semakin besar membutuhkan lebih banyak memori)'
    )
    batch_size_input.layout = widgets.Layout(width='100%', margin='2px 0')
    
    processing_section = widgets.VBox([
        widgets.HTML(value="<div style='font-weight:bold;color:#FF9800;margin-bottom:6px;'>⚡ Pemrosesan</div>"),
        target_splits_select,
        batch_size_input
    ])
    processing_section.layout = widgets.Layout(width='48%', padding='8px')
    
    # === VALIDATION SECTION ===
    
    validation_checkbox = widgets.Checkbox(
        value=validation_config.get('enabled', False),
        description='Validasi Dataset (Minimal)',
        style={'description_width': 'initial'},
        tooltip='Aktifkan validasi dataset untuk memeriksa kualitas data'
    )
    validation_checkbox.layout = widgets.Layout(width='100%', margin='2px 0')
    
    move_invalid_checkbox = widgets.Checkbox(
        value=preprocessing_config.get('move_invalid', False),
        description='Pindahkan File Tidak Valid',
        style={'description_width': 'initial'},
        tooltip='Pindahkan file yang tidak valid ke direktori terpisah'
    )
    move_invalid_checkbox.layout = widgets.Layout(width='100%', margin='2px 0')
    
    invalid_dir_input = widgets.Text(
        value=preprocessing_config.get('invalid_dir', 'data/invalid'),
        description='Direktori Invalid:',
        style={'description_width': '90px'},
        placeholder='Masukkan direktori untuk file tidak valid',
        tooltip='Lokasi penyimpanan file yang tidak lolos validasi'
    )
    invalid_dir_input.layout = widgets.Layout(width='100%', margin='2px 0')
    
    validation_section = widgets.VBox([
        widgets.HTML(value="<div style='font-weight:bold;color:#4CAF50;margin-bottom:6px;'>✅ Validasi</div>"),
        validation_checkbox,
        move_invalid_checkbox,
        invalid_dir_input
    ])
    validation_section.layout = widgets.Layout(width='48%', padding='8px')
    
    # === CLEANUP SECTION ===
    
    cleanup_target_options = [(target.value.title(), target.value) for target in CleanupTarget]
    cleanup_target_dropdown = widgets.Dropdown(
        options=cleanup_target_options,
        value=preprocessing_config.get('cleanup_target', CleanupTarget.PREPROCESSED.value),
        description='Target:',
        style={'description_width': '90px'},
        tooltip='Pilih target yang akan dibersihkan'
    )
    cleanup_target_dropdown.layout = widgets.Layout(width='100%', margin='2px 0')
    
    backup_checkbox = widgets.Checkbox(
        value=preprocessing_config.get('backup_enabled', True),
        description='Buat Backup Sebelum Hapus',
        style={'description_width': 'initial'},
        tooltip='Buat cadangan data sebelum melakukan pembersihan'
    )
    backup_checkbox.layout = widgets.Layout(width='100%', margin='2px 0')
    
    cleanup_section = widgets.VBox([
        widgets.HTML(value="<div style='font-weight:bold;color:#F44336;margin-bottom:6px;'>🧹 Pembersihan</div>"),
        cleanup_target_dropdown,
        backup_checkbox
    ])
    cleanup_section.layout = widgets.Layout(width='48%', padding='8px')
    
    # === LAYOUT ASSEMBLY ===
    
    top_row = widgets.HBox([normalization_section, processing_section])
    top_row.layout = widgets.Layout(width='100%', justify_content='space-between')
    
    bottom_row = widgets.HBox([validation_section, cleanup_section])
    bottom_row.layout = widgets.Layout(width='100%', justify_content='space-between')
    
    options_container = widgets.VBox([
        widgets.HTML(value="<h5 style='margin:8px 0;color:#495057;border-bottom:2px solid #28a745;padding-bottom:4px;'>⚙️ Konfigurasi Pra-pemrosesan</h5>"),
        top_row,
        bottom_row,
        widgets.HTML(value="<div style='margin-top:10px;font-size:12px;color:#6c757d;'>"
                         "<i>Pastikan konfigurasi sesuai sebelum memulai proses</i></div>")
    ])
    options_container.layout = widgets.Layout(
        padding='12px', 
        width='100%', 
        border='1px solid #dee2e6',
        border_radius='6px', 
        background_color='#f8f9fa',
        margin='5px 0 15px 0'
    )
    
    # Attach components for access from parent
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.preserve_aspect_checkbox = preserve_aspect_checkbox
    options_container.target_splits_select = target_splits_select
    options_container.batch_size_input = batch_size_input
    options_container.validation_checkbox = validation_checkbox
    options_container.move_invalid_checkbox = move_invalid_checkbox
    options_container.invalid_dir_input = invalid_dir_input
    options_container.cleanup_target_dropdown = cleanup_target_dropdown
    options_container.backup_checkbox = backup_checkbox
    
    return options_container
