"""
File: smartcash/ui/model/backbone/components/model_form.py
Deskripsi: Form component untuk model configuration dengan enhanced styling
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_model_form(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """Create model configuration form dengan styling konsisten
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        VBox containing the model form
    """
    config = config or {}
    model_config = config.get('model', {})
    
    # === STYLES ===
    
    dropdown_style = {'description_width': '140px'}
    dropdown_layout = widgets.Layout(width='100%', margin='5px 0')
    
    checkbox_layout = widgets.Layout(width='100%', margin='5px 0')
    
    section_header_style = """
        <h4 style="
            margin: 15px 0 10px 0;
            color: #2c3e50;
            font-weight: 600;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        ">{}</h4>
    """
    
    help_text_style = """
        <small style="
            color: #7f8c8d;
            display: block;
            margin: 3px 0 10px 20px;
            font-style: italic;
        ">{}</small>
    """
    
    # === PEMILIHAN BACKBONE ===
    
    backbone_dropdown = widgets.Dropdown(
        options=[
            ('Pilih Backbone', ''),
            ('EfficientNet-B4 (Recommended)', 'efficientnet_b4'),
            ('CSPDarknet (YOLOv5 Baseline)', 'cspdarknet')
        ],
        value=model_config.get('backbone', 'efficientnet_b4'),
        description='Backbone:',
        style=dropdown_style,
        layout=dropdown_layout
    )
    
    # === DETECTION CONFIGURATION ===
    
    detection_layers_select = widgets.SelectMultiple(
        options=[
            ('Banknote Detection', 'banknote'),
            ('Nominal Recognition', 'nominal'),
            ('Security Features', 'security')
        ],
        value=tuple(model_config.get('detection_layers', ['banknote'])),
        description='Detection Layers:',
        rows=3,
        style=dropdown_style,
        layout=widgets.Layout(width='100%', margin='5px 0', height='80px')
    )
    
    layer_mode_dropdown = widgets.Dropdown(
        options=[
            ('Satu Lapisan (Cepat)', 'single'),
            ('Multi Lapisan (Akurat)', 'multilayer')
        ],
        value=model_config.get('layer_mode', 'single'),
        description='Mode Lapisan:',
        style=dropdown_style,
        layout=dropdown_layout
    )
    
    # === OPTIMIZATION OPTIONS ===
    # Optimasi fitur
    feature_optimization_checkbox = widgets.Checkbox(
        value=model_config.get('feature_optimization', {}).get('enabled', False),
        description='Aktifkan Optimasi Fitur',
        indent=False,
        style={'description_width': 'auto'},
        layout=checkbox_layout
    )
    
    # Pelatihan presisi campuran
    mixed_precision_checkbox = widgets.Checkbox(
        value=model_config.get('mixed_precision', True),
        description='Aktifkan Pelatihan Presisi Campuran',
        indent=False,
        style={'description_width': 'auto'},
        layout=checkbox_layout
    )
    
    # === SECTION ORGANIZATION ===
    
    # Bagian Arsitektur
    architecture_section = widgets.VBox([
        widgets.HTML(section_header_style.format('🏗️ Arsitektur Model')),
        backbone_dropdown,
        widgets.HTML(help_text_style.format('Jaringan backbone untuk ekstraksi fitur'))
    ], layout=widgets.Layout(margin='0 0 10px 0'))
    
    # Bagian Deteksi
    detection_section = widgets.VBox([
        widgets.HTML(section_header_style.format('🎯 Konfigurasi Deteksi')),
        detection_layers_select,
        widgets.HTML(help_text_style.format('Pilih layer deteksi yang dibutuhkan')),
        layer_mode_dropdown,
        widgets.HTML(help_text_style.format('Mode operasi layer deteksi'))
    ], layout=widgets.Layout(margin='0 0 10px 0'))
    
    # Bagian Optimasi
    optimization_section = widgets.VBox([
        widgets.HTML(section_header_style.format('⚡ Optimasi')),
        feature_optimization_checkbox,
        widgets.HTML(help_text_style.format('Channel attention untuk akurasi yang lebih baik')),
        mixed_precision_checkbox,
        widgets.HTML(help_text_style.format('Pelatihan lebih cepat dengan presisi yang lebih rendah'))
    ], layout=widgets.Layout(margin='0'))
    
    # === MAIN FORM ASSEMBLY ===
    
    form = widgets.VBox([
        architecture_section,
        detection_section,
        optimization_section
    ], layout=widgets.Layout(
        width='58%',  # Slightly wider for better proportion
        padding='20px',
        border='1px solid #ddd',
        border_radius='10px',
        background_color='#ffffff',
        box_shadow='0 2px 8px rgba(0,0,0,0.08)'
    ))
    
    # Store widgets as attributes untuk easy access
    form.backbone_dropdown = backbone_dropdown
    form.detection_layers_select = detection_layers_select
    form.layer_mode_dropdown = layer_mode_dropdown
    form.feature_optimization_checkbox = feature_optimization_checkbox
    form.mixed_precision_checkbox = mixed_precision_checkbox
    
    # Add change observers untuk validation feedback
    _add_validation_observers(form)
    
    return form

def _add_validation_observers(form: widgets.VBox) -> None:
    """Add validation observers to form widgets"""
    
    def validate_layer_compatibility(change):
        """Validate layer mode dengan selected layers"""
        if hasattr(form, 'layer_mode_dropdown') and hasattr(form, 'detection_layers_select'):
            if form.layer_mode_dropdown.value == 'single' and len(form.detection_layers_select.value) > 1:
                form.layer_mode_dropdown.style = {'description_width': '140px', 'description_color': '#e74c3c'}
            else:
                form.layer_mode_dropdown.style = {'description_width': '140px'}
    
    # Add observers
    if hasattr(form, 'detection_layers_select'):
        form.detection_layers_select.observe(validate_layer_compatibility, names='value')
    if hasattr(form, 'layer_mode_dropdown'):
        form.layer_mode_dropdown.observe(validate_layer_compatibility, names='value')

def update_form_values(form: widgets.VBox, config: Dict[str, Any]) -> None:
    """Update form values dari configuration
    
    Args:
        form: Form widget
        config: Configuration dictionary
    """
    model_config = config.get('model', {})
    
    if hasattr(form, 'backbone_dropdown'):
        form.backbone_dropdown.value = model_config.get('backbone', 'efficientnet_b4')
    
    if hasattr(form, 'detection_layers_select'):
        form.detection_layers_select.value = tuple(model_config.get('detection_layers', ['banknote']))
    
    if hasattr(form, 'layer_mode_dropdown'):
        form.layer_mode_dropdown.value = model_config.get('layer_mode', 'single')
    
    if hasattr(form, 'feature_optimization_checkbox'):
        form.feature_optimization_checkbox.value = model_config.get('feature_optimization', {}).get('enabled', False)
    
    if hasattr(form, 'mixed_precision_checkbox'):
        form.mixed_precision_checkbox.value = model_config.get('mixed_precision', True)