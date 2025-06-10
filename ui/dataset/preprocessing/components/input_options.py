"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Enhanced form components dengan multi-split, validasi, aspect ratio, dan styling optimal
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """Enhanced form dengan multi-split, validasi, aspect ratio, dan styling optimal"""
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    normalization_config = preprocessing_config.get('normalization', {})
    validation_config = preprocessing_config.get('validation', {})
    performance_config = config.get('performance', {})
    
    # === SECTION 1: DATA & FORMAT ===
    
    # Resolution dropdown
    target_size = normalization_config.get('target_size', [640, 640])
    resolution_str = f"{target_size[0]}x{target_size[1]}" if isinstance(target_size, list) and len(target_size) >= 2 else "640x640"
    
    resolution_dropdown = widgets.Dropdown(
        options=['320x320', '416x416', '512x512', '640x640', '832x832'],
        value=resolution_str if resolution_str in ['320x320', '416x416', '512x512', '640x640', '832x832'] else '640x640',
        description='Resolusi Output:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Normalization dropdown
    normalization_raw = normalization_config.get('method', 'minmax')
    normalization_enabled = normalization_config.get('enabled', True)
    normalization_value = 'none' if not normalization_enabled else normalization_raw
    
    normalization_dropdown = widgets.Dropdown(
        options=[('Min-Max (0-1)', 'minmax'), ('Standard (z-score)', 'standard'), ('Tanpa Normalisasi', 'none')],
        value=normalization_value if normalization_value in ['minmax', 'standard', 'none'] else 'minmax',
        description='Normalisasi:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Preserve aspect ratio checkbox
    preserve_aspect_checkbox = widgets.Checkbox(
        value=normalization_config.get('preserve_aspect_ratio', True),
        description='Pertahankan Aspect Ratio',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='4px 0')
    )
    
    data_format_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#495057;margin-bottom:6px;font-size:14px;'>🖼️ Data & Format</div>"),
        resolution_dropdown,
        normalization_dropdown,
        preserve_aspect_checkbox
    ], layout=widgets.Layout(width='48%', padding='8px'))
    
    # === SECTION 2: TARGET SPLITS ===
    
    # Multi-select untuk target splits
    target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
    if isinstance(target_splits, str):
        target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
    
    target_splits_select = widgets.SelectMultiple(
        options=[('Training Set', 'train'), ('Validation Set', 'valid'), ('Test Set', 'test')],
        value=tuple(target_splits) if isinstance(target_splits, list) else ('train', 'valid'),
        description='Target Splits:',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='100%', height='80px', margin='2px 0')
    )
    
    # Batch size input
    batch_size_input = widgets.BoundedIntText(
        value=performance_config.get('batch_size', 32),
        min=1,
        max=128,
        step=1,
        description='Batch Size:',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    splits_performance_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#495057;margin-bottom:6px;font-size:14px;'>🎯 Target & Performance</div>"),
        target_splits_select,
        batch_size_input
    ], layout=widgets.Layout(width='48%', padding='8px'))
    
    # === SECTION 3: VALIDATION SETTINGS ===
    
    # Validation enabled checkbox
    validation_checkbox = widgets.Checkbox(
        value=validation_config.get('enabled', True),
        description='Aktifkan Validasi Dataset',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Move invalid checkbox
    move_invalid_checkbox = widgets.Checkbox(
        value=validation_config.get('move_invalid', True),
        description='Pindahkan File Invalid',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Invalid directory input
    invalid_dir_input = widgets.Text(
        value=validation_config.get('invalid_dir', 'data/invalid'),
        description='Lokasi Invalid:',
        placeholder='data/invalid',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    validation_section = widgets.VBox([
        widgets.HTML("<div style='font-weight:bold;color:#495057;margin-bottom:6px;font-size:14px;'>✅ Validasi Dataset</div>"),
        validation_checkbox,
        move_invalid_checkbox,
        invalid_dir_input
    ], layout=widgets.Layout(width='100%', padding='8px'))
    
    # === LAYOUT ASSEMBLY ===
    
    # Top row: Data/Format + Splits/Performance
    top_row = widgets.HBox([
        data_format_section,
        splits_performance_section
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='flex-start'
    ))
    
    # Main container dengan optimized styling
    options_container = widgets.VBox([
        widgets.HTML("<h5 style='margin:8px 0;color:#495057;border-bottom:2px solid #28a745;padding-bottom:4px;'>⚙️ Konfigurasi Preprocessing</h5>"),
        top_row,
        validation_section
    ], layout=widgets.Layout(
        padding='12px',
        width='100%',
        max_width='100%',
        border='1px solid #dee2e6',
        border_radius='6px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # Attach references untuk akses mudah
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.preserve_aspect_checkbox = preserve_aspect_checkbox
    options_container.target_splits_select = target_splits_select
    options_container.batch_size_input = batch_size_input
    options_container.validation_checkbox = validation_checkbox
    options_container.move_invalid_checkbox = move_invalid_checkbox
    options_container.invalid_dir_input = invalid_dir_input
    
    return options_container