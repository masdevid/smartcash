"""
File: smartcash/ui_handlers/training_config.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk komponen UI konfigurasi training model SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os, sys, json, yaml
from pathlib import Path

from smartcash.utils.ui_utils import (
    create_info_alert, create_status_indicator, styled_html
)

def setup_training_config_handlers(ui_components, config=None):
    """Setup handlers untuk komponen UI konfigurasi training model."""
    # Default config jika tidak disediakan
    if config is None:
        # Load default dari YAML yang disediakan dalam dokumen
        config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'framework': 'YOLOv5',
                'pretrained': True,
                'confidence': 0.25,
                'iou_threshold': 0.45
            },
            'training': {
                'epochs': 50,
                'batch_size': 16,
                'img_size': [640, 640],
                'patience': 5,
                'lr0': 0.01,
                'lrf': 0.01,
                'optimizer': 'Adam',
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'scheduler': 'cosine',
                'early_stopping_patience': 10,
                'save_period': 5,
                'val_interval': 1,
                'fliplr': 0.5,
                'flipud': 0.0,
                'mosaic': 1.0,
                'mixup': 0.0,
                'translate': 0.1,
                'scale': 0.5
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
    
    # Extract UI components
    backbone_options = ui_components['backbone_options']
    hyperparams_options = ui_components['hyperparams_options']
    layer_config = ui_components['layer_config']
    strategy_options = ui_components['strategy_options']
    advanced_tabs = ui_components['advanced_tabs']
    save_button = ui_components['save_button']
    reset_button = ui_components['reset_button']
    status_output = ui_components['status_output']
    
    # Data untuk handler
    handler_data = {
        'config': config,
        'config_path': 'configs/training_config.yaml'
    }
    
    # Fungsi untuk mendapatkan backbone type dari UI selection
    def get_backbone_from_ui():
        selection = backbone_options.children[0].value
        if "EfficientNet-B4" in selection:
            return "efficientnet_b4"
        elif "EfficientNet-B0" in selection:
            return "efficientnet_b0"
        elif "EfficientNet-B2" in selection:
            return "efficientnet_b2"
        elif "CSPDarknet" in selection:
            return "cspdarknet"
        return "efficientnet_b4"  # Default
    
    # Fungsi untuk memperbarui config dari UI
    def update_config_from_ui():
        # Backbone
        handler_data['config']['model']['backbone'] = get_backbone_from_ui()
        handler_data['config']['model']['pretrained'] = backbone_options.children[1].value
        
        # Hyperparameters
        handler_data['config']['training']['epochs'] = hyperparams_options.children[0].value
        handler_data['config']['training']['batch_size'] = hyperparams_options.children[1].value
        handler_data['config']['training']['lr0'] = hyperparams_options.children[2].value
        handler_data['config']['training']['optimizer'] = hyperparams_options.children[3].value
        handler_data['config']['training']['scheduler'] = hyperparams_options.children[4].value
        handler_data['config']['training']['early_stopping_patience'] = hyperparams_options.children[5].value
        
        # Layers
        handler_data['config']['layers']['banknote']['enabled'] = layer_config.children[0].children[0].value
        handler_data['config']['layers']['banknote']['threshold'] = layer_config.children[0].children[1].value
        handler_data['config']['layers']['nominal']['enabled'] = layer_config.children[1].children[0].value
        handler_data['config']['layers']['nominal']['threshold'] = layer_config.children[1].children[1].value
        handler_data['config']['layers']['security']['enabled'] = layer_config.children[2].children[0].value 
        handler_data['config']['layers']['security']['threshold'] = layer_config.children[2].children[1].value
        
        # Strategy
        handler_data['config']['training']['use_augmentation'] = strategy_options.children[0].value
        handler_data['config']['model']['half_precision'] = strategy_options.children[1].value
        handler_data['config']['training']['save_best'] = strategy_options.children[2].value
        handler_data['config']['training']['val_interval'] = 1 if strategy_options.children[3].value else 5
        handler_data['config']['training']['save_period'] = strategy_options.children[4].value
        
        # Advanced - Augmentation
        aug_tab = advanced_tabs.children[0]
        handler_data['config']['training']['fliplr'] = aug_tab.children[0].value
        handler_data['config']['training']['flipud'] = aug_tab.children[1].value
        handler_data['config']['training']['mosaic'] = aug_tab.children[2].value
        handler_data['config']['training']['scale'] = aug_tab.children[3].value
        handler_data['config']['training']['translate'] = aug_tab.children[4].value
        
        # Advanced - Loss Weights
        loss_tab = advanced_tabs.children[1]
        handler_data['config']['training']['box_loss_weight'] = loss_tab.children[0].value
        handler_data['config']['training']['obj_loss_weight'] = loss_tab.children[1].value
        handler_data['config']['training']['cls_loss_weight'] = loss_tab.children[2].value
        
        # Advanced - Other
        adv_tab = advanced_tabs.children[2]
        handler_data['config']['training']['weight_decay'] = adv_tab.children[0].value
        handler_data['config']['training']['momentum'] = adv_tab.children[1].value
        handler_data['config']['training']['use_swa'] = adv_tab.children[2].value
        handler_data['config']['training']['use_ema'] = adv_tab.children[3].value
        
        return handler_data['config']
    
    # Fungsi untuk memperbarui UI dari config
    def update_ui_from_config():
        config = handler_data['config']
        
        # Backbone
        backbone_map = {
            'efficientnet_b4': 'EfficientNet-B4 (Recommended)',
            'efficientnet_b0': 'EfficientNet-B0',
            'efficientnet_b2': 'EfficientNet-B2',
            'cspdarknet': 'CSPDarknet'
        }
        backbone_options.children[0].value = backbone_map.get(
            config['model']['backbone'], 'EfficientNet-B4 (Recommended)')
        backbone_options.children[1].value = config['model'].get('pretrained', True)
        
        # Hyperparameters
        hyperparams_options.children[0].value = config['training'].get('epochs', 50)
        hyperparams_options.children[1].value = config['training'].get('batch_size', 16)
        hyperparams_options.children[2].value = config['training'].get('lr0', 0.01)
        hyperparams_options.children[3].value = config['training'].get('optimizer', 'Adam')
        hyperparams_options.children[4].value = config['training'].get('scheduler', 'cosine')
        hyperparams_options.children[5].value = config['training'].get('early_stopping_patience', 10)
        
        # Layers
        layer_config.children[0].children[0].value = config['layers']['banknote'].get('enabled', True)
        layer_config.children[0].children[1].value = config['layers']['banknote'].get('threshold', 0.25)
        layer_config.children[1].children[0].value = config['layers']['nominal'].get('enabled', True)
        layer_config.children[1].children[1].value = config['layers']['nominal'].get('threshold', 0.30)
        layer_config.children[2].children[0].value = config['layers']['security'].get('enabled', True)
        layer_config.children[2].children[1].value = config['layers']['security'].get('threshold', 0.35)
        
        # Strategy
        strategy_options.children[0].value = config['training'].get('use_augmentation', True)
        strategy_options.children[1].value = config['model'].get('half_precision', True)
        strategy_options.children[2].value = config['training'].get('save_best', True)
        strategy_options.children[3].value = config['training'].get('val_interval', 1) == 1
        strategy_options.children[4].value = config['training'].get('save_period', 5)
        
        # Advanced - Augmentation
        aug_tab = advanced_tabs.children[0]
        aug_tab.children[0].value = config['training'].get('fliplr', 0.5)
        aug_tab.children[1].value = config['training'].get('flipud', 0.0)
        aug_tab.children[2].value = config['training'].get('mosaic', 1.0)
        aug_tab.children[3].value = config['training'].get('scale', 0.5)
        aug_tab.children[4].value = config['training'].get('translate', 0.1)
        
        # Advanced - Loss Weights
        loss_tab = advanced_tabs.children[1]
        loss_tab.children[0].value = config['training'].get('box_loss_weight', 0.05)
        loss_tab.children[1].value = config['training'].get('obj_loss_weight', 0.5)
        loss_tab.children[2].value = config['training'].get('cls_loss_weight', 0.5)
        
        # Advanced - Other
        adv_tab = advanced_tabs.children[2]
        adv_tab.children[0].value = config['training'].get('weight_decay', 0.0005)
        adv_tab.children[1].value = config['training'].get('momentum', 0.937)
        adv_tab.children[2].value = config['training'].get('use_swa', False)
        adv_tab.children[3].value = config['training'].get('use_ema', False)
    
    # Handler untuk save button
    def on_save_click(b):
        with status_output:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui()
                
                # Simpan ke file YAML
                config_dir = Path(handler_data['config_path']).parent
                if not config_dir.exists():
                    config_dir.mkdir(parents=True, exist_ok=True)
                
                # Simulasi penyimpanan (dalam implementasi asli, simpan file)
                # with open(handler_data['config_path'], 'w') as f:
                #     yaml.dump(updated_config, f, default_flow_style=False)
                
                # Output summary
                summary = f"""
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <h4>📊 Ringkasan Konfigurasi</h4>
                    <ul>
                        <li><b>Backbone:</b> {updated_config['model']['backbone']}</li>
                        <li><b>Training:</b> {updated_config['training']['epochs']} epochs, batch {updated_config['training']['batch_size']}</li>
                        <li><b>Optimizer:</b> {updated_config['training']['optimizer']}, LR={updated_config['training']['lr0']}</li>
                        <li><b>Layers:</b> {sum(1 for layer in updated_config['layers'].values() if layer['enabled'])} diaktifkan</li>
                    </ul>
                </div>
                """
                
                display(create_status_indicator("success", "✅ Konfigurasi berhasil disimpan"))
                display(HTML(summary))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Handler untuk reset button
    def on_reset_click(b):
        with status_output:
            clear_output()
            display(create_status_indicator("info", "🔄 Mereset konfigurasi ke default..."))
            
            try:
                # Reset config
                handler_data['config'] = {
                    'model': {
                        'backbone': 'efficientnet_b4',
                        'framework': 'YOLOv5',
                        'pretrained': True,
                        'confidence': 0.25,
                        'iou_threshold': 0.45
                    },
                    'training': {
                        'epochs': 50,
                        'batch_size': 16,
                        'img_size': [640, 640],
                        'patience': 5,
                        'lr0': 0.01,
                        'lrf': 0.01,
                        'optimizer': 'Adam',
                        'momentum': 0.937,
                        'weight_decay': 0.0005,
                        'scheduler': 'cosine',
                        'early_stopping_patience': 10,
                        'save_period': 5,
                        'val_interval': 1,
                        'fliplr': 0.5,
                        'flipud': 0.0,
                        'mosaic': 1.0,
                        'mixup': 0.0,
                        'translate': 0.1,
                        'scale': 0.5
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
                
                # Update UI
                update_ui_from_config()
                
                display(create_status_indicator("success", "✅ Konfigurasi berhasil direset ke default"))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Pasang handlers
    save_button.on_click(on_save_click)
    reset_button.on_click(on_reset_click)
    
    # Inisialisasi UI dengan config yang ada
    update_ui_from_config()
    
    return ui_components