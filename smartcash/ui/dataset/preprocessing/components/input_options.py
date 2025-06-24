"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Essential preprocessing input forms sesuai API requirements
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """Create essential preprocessing forms dengan API compatibility"""
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    validation_config = preprocessing_config.get('validation', {})
    cleanup_config = preprocessing_config.get('cleanup', {})
    performance_config = config.get('performance', {})
    
    # === NORMALIZATION SECTION ===
    
    # Resolution dropdown
    target_size = normalization_config.get('target_size', [640, 640])
    resolution_str = f"{target_size[0]}x{target_size[1]}" if isinstance(target_size, list) else "640x640"
    
    resolution_dropdown = widgets.Dropdown(
        options=['320x320', '416x416', '512x512', '640x640', '832x832'],
        value=resolution_str if resolution_str in ['320x320', '416x416', '512x512', '640x640', '832x832'] else '640x640',
        description='Resolusi:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Normalization method
    normalization_enabled = normalization_config.get('enabled', True)
    method = normalization_config.get('method', 'minmax')
    current_value = 'none' if not normalization_enabled else method
    
    normalization_dropdown = widgets.Dropdown(
        options=[('Min-Max (0-1)', 'minmax'), ('Standard Z-Score', 'standard'), ('Tanpa Normalisasi', 'none')],
        value=current_value if current_value in ['minmax', 'standard', 'none'] else 'minmax',
        description='Metode:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Preserve aspect ratio
    preserve_aspect_checkbox = widgets.Checkbox(
        value=normalization_config.get('preserve_aspect_ratio', True),
        description='Pertahankan Aspect Ratio (YOLO)',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    normalization_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#2196F3;margin-bottom:6px;'>🎨 Normalisasi YOLO</div>"),
        resolution_dropdown,
        normalization_dropdown, 
        preserve_aspect_checkbox
    ], layout=widgets.Layout(width='48%', padding='8px'))
    
    # === PROCESSING SECTION ===
    
    # Target splits
    target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
    target_splits_select = widgets.SelectMultiple(
        options=[('Training', 'train'), ('Validation', 'valid'), ('Test', 'test')],
        value=tuple(target_splits) if isinstance(target_splits, list) else ('train', 'valid'),
        description='Target:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', height='70px', margin='2px 0')
    )
    
    # Batch size
    batch_size_input = widgets.BoundedIntText(
        value=performance_config.get('batch_size', 32),
        min=1, max=128, step=1,
        description='Batch Size:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    processing_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#FF9800;margin-bottom:6px;'>⚡ Processing</div>"),
        target_splits_select,
        batch_size_input
    ], layout=widgets.Layout(width='48%', padding='8px'))
    
    # === VALIDATION SECTION ===
    
    validation_checkbox = widgets.Checkbox(
        value=validation_config.get('enabled', True),
        description='Validasi Dataset',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    move_invalid_checkbox = widgets.Checkbox(
        value=validation_config.get('move_invalid', True),
        description='Pindahkan File Invalid',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    invalid_dir_input = widgets.Text(
        value=validation_config.get('invalid_dir', 'data/invalid'),
        description='Dir Invalid:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    validation_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#4CAF50;margin-bottom:6px;'>✅ Validasi</div>"),
        validation_checkbox,
        move_invalid_checkbox,
        invalid_dir_input
    ], layout=widgets.Layout(width='48%', padding='8px'))
    
    # === CLEANUP SECTION ===
    
    cleanup_target_dropdown = widgets.Dropdown(
        options=[('Data Preprocessed', 'preprocessed'), ('Sample Images', 'samples'), ('Keduanya', 'both')],
        value=cleanup_config.get('target', 'preprocessed'),
        description='Target:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    backup_checkbox = widgets.Checkbox(
        value=cleanup_config.get('backup_enabled', False),
        description='Buat Backup Sebelum Hapus',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    cleanup_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#F44336;margin-bottom:6px;'>🧹 Cleanup</div>"),
        cleanup_target_dropdown,
        backup_checkbox
    ], layout=widgets.Layout(width='48%', padding='8px'))
    
    # === LAYOUT ASSEMBLY ===
    
    top_row = widgets.HBox([normalization_section, processing_section], 
        layout=widgets.Layout(width='100%', justify_content='space-between'))
    
    bottom_row = widgets.HBox([validation_section, cleanup_section],
        layout=widgets.Layout(width='100%', justify_content='space-between'))
    
    options_container = widgets.VBox([
        widgets.HTML("<h5 style='margin:8px 0;color:#495057;border-bottom:2px solid #28a745;padding-bottom:4px;'>⚙️ Konfigurasi Preprocessing</h5>"),
        top_row,
        bottom_row
    ], layout=widgets.Layout(
        padding='12px', width='100%', border='1px solid #dee2e6',
        border_radius='6px', background_color='#f8f9fa'
    ))
    
    # Attach components untuk akses dari parent
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