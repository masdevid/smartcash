"""
File: smartcash/ui/training/components/config_tabs.py
Deskripsi: Fixed komponen tabs konfigurasi tanpa refresh button internal dan proper titles handling
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple

def create_config_tabs(config: Dict[str, Any]) -> widgets.Tab:
    """Create tabbed configuration display dengan konten dari YAML - tanpa refresh button internal"""
    
    # Extract config sections dengan safe defaults
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    hyperparameters = config.get('hyperparameters', {})
    training_utils = config.get('training_utils', {})
    paths_config = config.get('paths', {})
    
    # Tab content dengan one-liner HTML generation
    tab_contents = [
        _create_model_tab_html(model_config, training_config),
        _create_hyperparams_tab_html(hyperparameters, training_config),
        _create_strategy_tab_html(config, training_utils),
        _create_paths_tab_html(paths_config, training_utils, config)
    ]
    
    # Create tabs dengan proper widget construction
    tab_widgets = [widgets.HTML(content) for content in tab_contents]
    tabs = widgets.Tab(children=tab_widgets)
    
    # Set titles dengan proper method - fix untuk 'titles' attribute error
    tab_titles = ['Model', 'Hyperparameters', 'Strategy', 'Paths']
    [tabs.set_title(i, title) for i, title in enumerate(tab_titles)]
    
    return tabs

def _create_model_tab_html(model_config: Dict[str, Any], training_config: Dict[str, Any]) -> str:
    """Generate model tab HTML dengan one-liner extraction"""
    return f"""
    <div style="padding: 15px;">
        <h5 style="margin: 0 0 15px 0; color: #1976d2;">🧠 Model Configuration</h5>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 13px;">
            <div>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><b>Type:</b> <span style="color: #1976d2;">{model_config.get('type', training_config.get('model_type', 'efficient_basic'))}</span></li>
                    <li><b>Backbone:</b> <span style="color: #1976d2;">{model_config.get('backbone', training_config.get('backbone', 'efficientnet_b4'))}</span></li>
                    <li><b>Pretrained:</b> <span style="color: {'#2e7d32' if model_config.get('backbone_pretrained', True) else '#d32f2f'};">{'✅' if model_config.get('backbone_pretrained', True) else '❌'}</span></li>
                </ul>
            </div>
            <div>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><b>Confidence:</b> <span style="color: #1976d2;">{model_config.get('confidence', 0.25)}</span></li>
                    <li><b>IoU Threshold:</b> <span style="color: #1976d2;">{model_config.get('iou_threshold', 0.45)}</span></li>
                    <li><b>Classes:</b> <span style="color: #1976d2;">{model_config.get('num_classes', training_config.get('num_classes', 7))}</span></li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px;">
            <b>Optimizations:</b> 
            <span style="color: {'#2e7d32' if model_config.get('use_attention') else '#666'};">Attention: {'✅' if model_config.get('use_attention') else '❌'}</span>,
            <span style="color: {'#2e7d32' if model_config.get('use_residual') else '#666'};">Residual: {'✅' if model_config.get('use_residual') else '❌'}</span>,
            <span style="color: {'#2e7d32' if model_config.get('use_ciou') else '#666'};">CIoU: {'✅' if model_config.get('use_ciou') else '❌'}</span>
        </div>
    </div>
    """

def _create_hyperparams_tab_html(hyperparams: Dict[str, Any], training_config: Dict[str, Any]) -> str:
    """Generate hyperparameters tab HTML"""
    return f"""
    <div style="padding: 15px;">
        <h5 style="margin: 0 0 15px 0; color: #6a1b9a;">⚙️ Hyperparameters</h5>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 13px;">
            <div>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><b>Epochs:</b> <span style="color: #6a1b9a;">{hyperparams.get('epochs', training_config.get('epochs', 100))}</span></li>
                    <li><b>Batch Size:</b> <span style="color: #6a1b9a;">{hyperparams.get('batch_size', training_config.get('batch_size', 16))}</span></li>
                    <li><b>Learning Rate:</b> <span style="color: #6a1b9a;">{hyperparams.get('learning_rate', training_config.get('learning_rate', 0.001))}</span></li>
                </ul>
            </div>
            <div>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><b>Weight Decay:</b> <span style="color: #6a1b9a;">{hyperparams.get('weight_decay', training_config.get('weight_decay', 0.0005))}</span></li>
                    <li><b>Image Size:</b> <span style="color: #6a1b9a;">{hyperparams.get('image_size', training_config.get('image_size', 640))}</span></li>
                    <li><b>Optimizer:</b> <span style="color: #6a1b9a;">{hyperparams.get('optimizer', training_config.get('optimizer', 'Adam'))}</span></li>
                </ul>
            </div>
        </div>
    </div>
    """

def _create_strategy_tab_html(config: Dict[str, Any], training_utils: Dict[str, Any]) -> str:
    """Generate strategy tab HTML"""
    return f"""
    <div style="padding: 15px;">
        <h5 style="margin: 0 0 15px 0; color: #2e7d32;">🚀 Training Strategy</h5>
        <div style="font-size: 13px;">
            <ul style="margin: 0; padding-left: 20px;">
                <li><b>Early Stopping:</b> <span style="color: {'#2e7d32' if config.get('early_stopping', {}).get('enabled', True) else '#d32f2f'};">{'✅ Enabled' if config.get('early_stopping', {}).get('enabled', True) else '❌ Disabled'}</span> (Patience: {config.get('early_stopping', {}).get('patience', 15)})</li>
                <li><b>Save Best:</b> <span style="color: {'#2e7d32' if config.get('save_best', {}).get('enabled', True) else '#d32f2f'};">{'✅ Enabled' if config.get('save_best', {}).get('enabled', True) else '❌ Disabled'}</span></li>
                <li><b>Mixed Precision:</b> <span style="color: {'#2e7d32' if training_utils.get('mixed_precision', True) else '#d32f2f'};">{'✅ Enabled' if training_utils.get('mixed_precision', True) else '❌ Disabled'}</span></li>
                <li><b>Multi-Scale:</b> <span style="color: {'#2e7d32' if config.get('multi_scale', True) else '#d32f2f'};">{'✅ Enabled' if config.get('multi_scale', True) else '❌ Disabled'}</span></li>
                <li><b>Layer Mode:</b> <span style="color: #2e7d32;">{training_utils.get('layer_mode', 'single').title()}</span></li>
                <li><b>Experiment:</b> <span style="color: #2e7d32;">{training_utils.get('experiment_name', 'efficientnet_b4_training')}</span></li>
            </ul>
        </div>
    </div>
    """

def _create_paths_tab_html(paths_config: Dict[str, Any], training_utils: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate paths tab HTML"""
    return f"""
    <div style="padding: 15px;">
        <h5 style="margin: 0 0 15px 0; color: #00695c;">📁 Paths & Validation</h5>
        <div style="font-size: 13px;">
            <ul style="margin: 0; padding-left: 20px;">
                <li><b>Checkpoint Dir:</b><br><span style="color: #00695c; font-family: monospace; font-size: 11px;">{training_utils.get('checkpoint_dir', '/content/runs/train/checkpoints')}</span></li>
                <li><b>Pretrained Models:</b><br><span style="color: #00695c; font-family: monospace; font-size: 11px;">{config.get('pretrained_models_path', '/content/drive/MyDrive/SmartCash/models')}</span></li>
                <li><b>Data Dir:</b><br><span style="color: #00695c; font-family: monospace; font-size: 11px;">{paths_config.get('data_dir', '/data/preprocessed')}</span></li>
                <li><b>Validation Frequency:</b> <span style="color: #00695c;">{config.get('validation', {}).get('frequency', 1)} epoch(s)</span></li>
                <li><b>Log Metrics Every:</b> <span style="color: #00695c;">{training_utils.get('log_metrics_every', 10)} batches</span></li>
            </ul>
        </div>
    </div>
    """

def update_config_tabs(tabs_widget: widgets.Tab, config: Dict[str, Any]) -> widgets.Tab:
    """Update existing tabs dengan config baru - fixed titles handling"""
    if not tabs_widget or not hasattr(tabs_widget, 'children'):
        return create_config_tabs(config)
    
    try:
        # Create new tabs dengan updated config
        new_tabs = create_config_tabs(config)
        
        # Preserve selected index dan update children dengan one-liner safety
        selected_index = getattr(tabs_widget, 'selected_index', 0)
        tabs_widget.children = new_tabs.children
        
        # Update titles dengan safe method - fix untuk attribute error
        tab_titles = ['Model', 'Hyperparameters', 'Strategy', 'Paths']
        [tabs_widget.set_title(i, title) for i, title in enumerate(tab_titles) if i < len(tabs_widget.children)]
        
        # Restore selected index dengan bounds checking
        tabs_widget.selected_index = min(selected_index, len(tabs_widget.children) - 1) if selected_index is not None else None
        
        return tabs_widget
        
    except Exception as e:
        # Return new tabs jika update gagal
        return create_config_tabs(config)

# One-liner utilities untuk config extraction
extract_model_summary = lambda config: f"{config.get('model', {}).get('backbone', 'efficientnet_b4')} ({config.get('model', {}).get('type', 'efficient_basic')})"
get_training_params = lambda config: f"{config.get('epochs', 100)} epochs, batch {config.get('batch_size', 16)}, lr {config.get('learning_rate', 0.001)}"
check_optimization_flags = lambda model_config: [f"{opt}: {'✅' if model_config.get(f'use_{opt}') else '❌'}" for opt in ['attention', 'residual', 'ciou']]