"""
File: smartcash/ui_handlers/backbone_selection.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI pemilihan backbone dan layer model SmartCash dengan integrasi visualization.
"""

import threading
import time
import os
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output, HTML
from pathlib import Path

from smartcash.utils.ui_utils import create_status_indicator, create_metric_display

def setup_backbone_selection_handlers(ui_components, config=None):
    """Setup handlers untuk UI pemilihan backbone dan layer model."""
    # Inisialisasi dependencies
    logger = None
    layer_config_manager = None
    model_manager = None
    observer_manager = None
    config_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.layer_config_manager import get_layer_config
        from smartcash.handlers.model import ModelManager
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.visualization import VisualizationHelper
        
        logger = get_logger("backbone_selection")
        layer_config_manager = get_layer_config(logger=logger)
        config_manager = get_config_manager(logger=logger)
        
        # Load config jika belum ada
        if not config or not isinstance(config, dict):
            config = config_manager.load_config(
                filename="configs/model_config.yaml",
                fallback_to_pickle=True
            ) or {
                'model': {
                    'backbone': 'efficientnet_b4',
                    'framework': 'YOLOv5',
                    'pretrained': True,
                    'confidence': 0.25,
                    'iou_threshold': 0.45
                },
                'layers': {
                    'banknote': {
                        'enabled': True,
                        'threshold': 0.25
                    },
                    'nominal': {
                        'enabled': True,
                        'threshold': 0.30
                    },
                    'security': {
                        'enabled': True,
                        'threshold': 0.35
                    }
                }
            }
        
        # Inisialisasi model manager
        model_manager = ModelManager(config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
        else:
            print(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Kelompok observer
    observer_group = "backbone_selection_observers"
    
    # Handler untuk save button
    def on_save_click(b):
        with ui_components['status_output']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi backbone dan layer..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui()
                
                # Simpan ke file
                if config_manager:
                    success = config_manager.save_config(
                        updated_config, 
                        "configs/model_config.yaml",
                        backup=True
                    )
                    
                    if success:
                        display(create_status_indicator(
                            "success", 
                            "✅ Konfigurasi model berhasil disimpan ke configs/model_config.yaml"
                        ))
                    else:
                        display(create_status_indicator(
                            "warning", 
                            "⚠️ Konfigurasi diupdate dalam memori, tetapi gagal menyimpan ke file"
                        ))
                else:
                    # Just update in-memory if config_manager not available
                    display(create_status_indicator(
                        "success", 
                        "✅ Konfigurasi model diupdate dalam memori"
                    ))
                
                # Create summary of settings
                backbone_type = updated_config['model']['backbone']
                pretrained = updated_config['model']['pretrained']
                layers_enabled = sum(1 for layer, settings in updated_config['layers'].items() 
                                    if settings.get('enabled', False))
                
                summary_html = f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <h4>📊 Model Configuration Summary</h4>
                    <ul>
                        <li><b>Backbone:</b> {backbone_type}</li>
                        <li><b>Pretrained:</b> {'Yes' if pretrained else 'No'}</li>
                        <li><b>Layers:</b> {layers_enabled} active layers</li>
                        <li><b>Default confidence:</b> {updated_config['model'].get('confidence', 0.25)}</li>
                    </ul>
                </div>
                """
                display(HTML(summary_html))
                
                # Visualize layer configuration after save
                visualize_layer_config()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat menyimpan konfigurasi: {str(e)}"))
    
    # Handler untuk reset button
    def on_reset_click(b):
        with ui_components['status_output']:
            clear_output()
            display(create_status_indicator("info", "🔄 Reset konfigurasi ke default..."))
            
            try:
                # Default config values
                default_config = {
                    'model': {
                        'backbone': 'efficientnet_b4',
                        'framework': 'YOLOv5',
                        'pretrained': True,
                        'confidence': 0.25,
                        'iou_threshold': 0.45
                    },
                    'layers': {
                        'banknote': {
                            'enabled': True,
                            'threshold': 0.25
                        },
                        'nominal': {
                            'enabled': True,
                            'threshold': 0.30
                        },
                        'security': {
                            'enabled': True,
                            'threshold': 0.35
                        }
                    }
                }
                
                # Update UI from default
                update_ui_from_config(default_config)
                
                # Update global config
                if config:
                    config.update(default_config)
                
                display(create_status_indicator("success", "✅ Konfigurasi berhasil direset ke default"))
                
                # Visualize default layer configuration
                visualize_layer_config()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat reset konfigurasi: {str(e)}"))
    
    # Fungsi untuk update config dari UI
    def update_config_from_ui():
        # Get backbone config
        backbone_options = ui_components['backbone_options']
        backbone_selection = backbone_options.children[0].value
        backbone_type = 'efficientnet_b4' if 'EfficientNet' in backbone_selection else 'cspdarknet'
        pretrained = backbone_options.children[1].value
        freeze_backbone = backbone_options.children[2].value
        
        # Get layer config
        layer_config = ui_components['layer_config']
        layers = {}
        layer_names = ['banknote', 'nominal', 'security']
        
        for i, layer_name in enumerate(layer_names):
            if i < len(layer_config.children):
                layer_row = layer_config.children[i]
                
                if len(layer_row.children) >= 2:
                    layers[layer_name] = {
                        'enabled': layer_row.children[0].value,
                        'threshold': layer_row.children[1].value
                    }
        
        # Update config
        updated_config = config.copy() if config else {}
        
        # Ensure model config exists
        if 'model' not in updated_config:
            updated_config['model'] = {}
        
        # Update model settings
        updated_config['model'].update({
            'backbone': backbone_type,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone
        })
        
        # Update layer settings
        updated_config['layers'] = layers
        
        return updated_config
    
    # Fungsi untuk update UI dari config
    def update_ui_from_config(config_to_use=None):
        if not config_to_use:
            config_to_use = config
            
        if not config_to_use:
            return
        
        # Backbone settings
        if 'model' in config_to_use:
            model_cfg = config_to_use['model']
            backbone_options = ui_components['backbone_options']
            
            # Set backbone type
            if 'backbone' in model_cfg:
                backbone_type = model_cfg['backbone']
                backbone_options.children[0].value = (
                    'EfficientNet-B4 (Recommended)' if backbone_type == 'efficientnet_b4' 
                    else 'CSPDarknet'
                )
            
            # Set pretrained option
            if 'pretrained' in model_cfg:
                backbone_options.children[1].value = model_cfg['pretrained']
            
            # Set freeze backbone option
            if 'freeze_backbone' in model_cfg:
                backbone_options.children[2].value = model_cfg['freeze_backbone']
        
        # Layer settings
        if 'layers' in config_to_use:
            layer_cfg = config_to_use['layers']
            layer_config = ui_components['layer_config']
            layer_names = ['banknote', 'nominal', 'security']
            
            for i, layer_name in enumerate(layer_names):
                if layer_name in layer_cfg and i < len(layer_config.children):
                    layer_row = layer_config.children[i]
                    layer_settings = layer_cfg[layer_name]
                    
                    if len(layer_row.children) >= 2:
                        # Set enabled state
                        if 'enabled' in layer_settings:
                            layer_row.children[0].value = layer_settings['enabled']
                        
                        # Set threshold
                        if 'threshold' in layer_settings:
                            layer_row.children[1].value = layer_settings['threshold']
    
    # Fungsi visualisasi layer configuration
    def visualize_layer_config():
        try:
            with ui_components['layer_viz']:
                clear_output(wait=True)
                
                cfg = update_config_from_ui()
                if not cfg or 'layers' not in cfg:
                    return
                
                # Extract layer info
                layers = cfg['layers']
                layer_names = []
                thresholds = []
                enabled_status = []
                
                for name, settings in layers.items():
                    layer_names.append(name.capitalize())
                    thresholds.append(settings.get('threshold', 0.25))
                    enabled_status.append(settings.get('enabled', False))
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot thresholds
                colors = ['green' if status else 'gray' for status in enabled_status]
                ax1.bar(layer_names, thresholds, color=colors, alpha=0.7)
                ax1.set_ylim(0, max(thresholds) * 1.2)
                ax1.set_ylabel('Confidence Threshold')
                ax1.set_title('Layer Detection Thresholds')
                
                # Add threshold values on top of bars
                for i, v in enumerate(thresholds):
                    ax1.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')
                
                # Plot class counts by layer
                if layer_config_manager:
                    class_counts = []
                    for name in ['banknote', 'nominal', 'security']:
                        layer_cfg = layer_config_manager.get_layer_config(name)
                        class_counts.append(len(layer_cfg.get('classes', [])))
                    
                    ax2.bar(layer_names, class_counts, color='royalblue', alpha=0.7)
                    ax2.set_ylabel('Number of Classes')
                    ax2.set_title('Classes per Layer')
                    
                    # Add class count values on top of bars
                    for i, v in enumerate(class_counts):
                        ax2.text(i, v + 0.3, str(v), ha='center', fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                display(plt.gcf())
                plt.close()
                
                # Tabular summary of layers
                table_html = "<h4>📋 Layer Configuration Summary</h4>"
                table_html += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
                table_html += "<tr style='background:#f2f2f2'><th>Layer</th><th>Status</th><th>Threshold</th><th>Classes</th></tr>"
                
                for name in ['banknote', 'nominal', 'security']:
                    # Get settings from updated config
                    settings = layers.get(name, {})
                    enabled = settings.get('enabled', False)
                    threshold = settings.get('threshold', 0.25)
                    
                    # Get class info if layer_config_manager available
                    class_names = ""
                    if layer_config_manager:
                        layer_cfg = layer_config_manager.get_layer_config(name)
                        class_names = ", ".join(layer_cfg.get('classes', [])[:3])
                        if len(layer_cfg.get('classes', [])) > 3:
                            class_names += ", ..."
                    
                    # Status color and icon
                    status_color = "green" if enabled else "gray"
                    status_icon = "✅" if enabled else "❌"
                    
                    table_html += f"<tr>"
                    table_html += f"<td>{name.capitalize()}</td>"
                    table_html += f"<td style='color:{status_color}'>{status_icon} {'Aktif' if enabled else 'Nonaktif'}</td>"
                    table_html += f"<td>{threshold:.2f}</td>"
                    table_html += f"<td>{class_names}</td>"
                    table_html += f"</tr>"
                
                table_html += "</table>"
                display(HTML(table_html))
                
                # Add info about model and backbone
                model_info = f"""
                <div style="margin-top:15px; padding:10px; background:#f8f9fa; border-radius:5px;">
                <p><b>🧠 Backbone:</b> {cfg['model'].get('backbone', 'efficientnet_b4')} {'(pretrained)' if cfg['model'].get('pretrained', True) else ''}</p>
                <p><b>Total Classes:</b> {sum(len(layer_config_manager.get_layer_config(name).get('classes', [])) for name in ['banknote', 'nominal', 'security']) if layer_config_manager else 'N/A'}</p>
                </div>
                """
                display(HTML(model_info))
                
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat visualisasi layer: {str(e)}")
            display(HTML(f"<p style='color:red'>❌ Error saat visualisasi: {str(e)}</p>"))
    
    # Register callbacks
    ui_components['save_button'].on_click(on_save_click)
    ui_components['reset_button'].on_click(on_reset_click)
    
    # Inisialisasi UI dari config
    update_ui_from_config()
    
    # Visualisasi awal
    visualize_layer_config()
    
    # Cleanup function
    def cleanup():
        """Bersihkan resources saat keluar dari scope."""
        if observer_manager:
            observer_manager.unregister_group(observer_group)
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components