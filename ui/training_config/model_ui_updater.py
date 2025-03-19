"""
File: smartcash/ui/training_config/model_ui_updater.py
Deskripsi: Pembaruan UI konfigurasi model dan visualisasi summary
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS
from smartcash.ui.training_config.model_config_definitions import get_model_config

def update_ui_for_model_type(ui_components: Dict[str, Any], model_type: str, config: Dict[str, Any]) -> None:
    """
    Update UI berdasarkan model type yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        model_type: Tipe model yang dipilih
        config: Konfigurasi yang akan diupdate
    """
    # Dapatkan konfigurasi model
    model_config = get_model_config(model_type)
    
    # Update backbone dropdown
    backbone_options = ui_components.get('backbone_options')
    if backbone_options and hasattr(backbone_options, 'children') and len(backbone_options.children) > 0:
        backbone_dropdown = backbone_options.children[0]
        
        for i, option in enumerate(backbone_dropdown.options):
            if option.startswith(model_config['backbone']):
                backbone_dropdown.index = i
                break
    
    # Update fitur options
    features_options = ui_components.get('features_options')
    if features_options and hasattr(features_options, 'children') and len(features_options.children) >= 4:
        # Update attention, residual, ciou checkboxes dan num_repeats slider
        features_options.children[0].value = model_config['use_attention']
        features_options.children[1].value = model_config['use_residual']
        features_options.children[2].value = model_config['use_ciou']
        features_options.children[3].value = model_config['num_repeats']
        
        # Aktifkan/nonaktifkan num_repeats berdasarkan use_residual
        features_options.children[3].disabled = not model_config['use_residual']
    
    # Update konfigurasi di memory
    if 'model' not in config:
        config['model'] = {}
        
    config['model']['model_type'] = model_type
    config['model'].update(model_config)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update seluruh UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi model
    """
    # Check if config exists
    if not config or not isinstance(config, dict):
        return
        
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('model_type', 'efficient_optimized')
    
    # Update model type dropdown
    model_options = ui_components.get('model_options')
    if not model_options or not hasattr(model_options, 'children') or len(model_options.children) < 1:
        return
        
    model_dropdown = model_options.children[0]
    
    if not hasattr(model_dropdown, 'options'):
        return
        
    for i, option in enumerate(model_dropdown.options):
        if option.startswith(model_type):
            model_dropdown.index = i
            break
                
    # Backbone settings sudah otomatis diupdate oleh model type change handler
                
    # Update layer settings
    if 'layers' in config:
        layer_cfg = config['layers']
        layer_config = ui_components.get('layer_config')
        layer_names = ['banknote', 'nominal', 'security']
        
        if not layer_config or not hasattr(layer_config, 'children'):
            return
            
        for i, layer_name in enumerate(layer_names):
            if layer_name in layer_cfg and i < len(layer_config.children):
                layer_row = layer_config.children[i]
                layer_settings = layer_cfg[layer_name]
                
                if hasattr(layer_row, 'children') and len(layer_row.children) >= 2:
                    # Set enabled & threshold
                    layer_row.children[0].value = layer_settings.get('enabled', True)
                    layer_row.children[1].value = layer_settings.get('threshold', 0.25)
    
    # Update summary
    update_layer_summary(ui_components, config)

def update_layer_summary(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update visualisasi summary layer dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi model
    """
    layer_summary = ui_components.get('layer_summary')
    if not layer_summary:
        return
        
    with layer_summary:
        clear_output(wait=True)
        
        try:
            # Dapatkan nilai dari konfigurasi model
            model_cfg = config.get('model', {})
            model_type = model_cfg.get('model_type', 'efficient_optimized')
            backbone_type = model_cfg.get('backbone', 'efficientnet_b4')
            
            # Dapatkan status fitur
            use_attention = model_cfg.get('use_attention', False)
            use_residual = model_cfg.get('use_residual', False)
            use_ciou = model_cfg.get('use_ciou', False)
            num_repeats = model_cfg.get('num_repeats', 1)
            
            # Buat HTML summary yang informatif
            html = f"<h4 style='margin-top:0; color:{COLORS['dark']}; margin-bottom:10px;'>📋 Model & Layer Configuration</h4>"
            
            # Tampilkan model info
            html += f"""
            <div style="margin-bottom:15px; padding:10px; background:#f8f9fa; border-radius:5px; color:{COLORS['dark']}">
                <p><b>🧠 Model:</b> {model_type}</p>
                <p><b>⚙️ Backbone:</b> {backbone_type}</p>
                <p><b>🔌 Features:</b> 
                    {'Attention' if use_attention else ''} 
                    {' + Residual' if use_residual else ''} 
                    {' + CIoU' if use_ciou else ''}
                    {' (None)' if not any([use_attention, use_residual, use_ciou]) else ''}
                </p>
                {f'<p><b>🔄 Residual Blocks:</b> {num_repeats}</p>' if use_residual else ''}
            </div>
            """
            
            # Tampilkan tabel untuk layer
            html += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
            html += "<tr style='background:#f2f2f2'><th>Layer</th><th>Status</th><th>Threshold</th></tr>"
            
            for name in ['banknote', 'nominal', 'security']:
                settings = config.get('layers', {}).get(name, {})
                enabled = settings.get('enabled', False)
                threshold = settings.get('threshold', 0.25)
                
                status_color = "green" if enabled else "gray"
                status_icon = "✅" if enabled else "❌"
                
                html += f"<tr>"
                html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{name.capitalize()}</td>"
                html += f"<td style='padding:8px; border-bottom:1px solid #ddd; color:{status_color}'>{status_icon} {'Aktif' if enabled else 'Nonaktif'}</td>"
                html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{threshold:.2f}</td>"
                html += f"</tr>"
            
            html += "</table>"
            
            # Tampilkan enabled layers
            enabled_layers = [name.capitalize() for name, settings in config.get('layers', {}).items() 
                             if settings.get('enabled', False)]
            
            html += f"""
            <div style="margin-top:15px; padding:10px; background:#f8f9fa; border-radius:5px; color:#2c3e50">
            <p><b>✓ Layer Aktif:</b> {', '.join(enabled_layers) if enabled_layers else 'Tidak ada layer aktif'}</p>
            </div>
            """
            
            display(HTML(html))
            
        except Exception as e:
            # Logger already checked in main handler
            display(HTML(f"<p style='color:red'>❌ Error: {str(e)}</p>"))