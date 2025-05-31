"""
File: smartcash/ui/training/components/config_tabs.py
Deskripsi: Komponen tabs konfigurasi untuk training UI
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple

def create_config_tabs(config: Dict[str, Any]) -> widgets.Tab:
    """Create tabbed configuration display untuk menghemat tempat"""
    from smartcash.ui.components.tab_factory import create_tab_widget as create_tabs
    
    # Extract config sections
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    hyperparameters = config.get('hyperparameters', {})
    paths_config = config.get('paths', {})
    
    # Tab 1: Model Configuration
    model_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #1976d2;">🧠 Model Configuration</h5>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
            <li><b>Model Type:</b> <span style="color: #1976d2;">{model_config.get('model_type', training_config.get('model_type', 'efficient_optimized'))}</span></li>
            <li><b>Backbone:</b> <span style="color: #1976d2;">{model_config.get('backbone', training_config.get('backbone', 'efficientnet_b4'))}</span></li>
            <li><b>Detection Layers:</b> <span style="color: #1976d2;">{', '.join(model_config.get('detection_layers', training_config.get('detection_layers', ['banknote'])))}</span></li>
            <li><b>Classes:</b> <span style="color: #1976d2;">{model_config.get('num_classes', training_config.get('num_classes', 7))}</span></li>
        </ul>
    </div>
    """
    
    # Tab 2: Hyperparameters
    hyperparams_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #6a1b9a;">⚙️ Hyperparameters</h5>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
            <li><b>Batch Size:</b> <span style="color: #6a1b9a;">{hyperparameters.get('batch_size', training_config.get('batch_size', 16))}</span></li>
            <li><b>Learning Rate:</b> <span style="color: #6a1b9a;">{hyperparameters.get('learning_rate', training_config.get('learning_rate', 0.001))}</span></li>
            <li><b>Epochs:</b> <span style="color: #6a1b9a;">{hyperparameters.get('epochs', training_config.get('epochs', 100))}</span></li>
            <li><b>Image Size:</b> <span style="color: #6a1b9a;">{hyperparameters.get('image_size', training_config.get('image_size', 640))}</span></li>
            <li><b>Optimizer:</b> <span style="color: #6a1b9a;">{hyperparameters.get('optimizer', {}).get('type', 'SGD')}</span></li>
        </ul>
    </div>
    """
    
    # Tab 3: Training Strategy
    strategy_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #2e7d32;">🚀 Training Strategy</h5>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
            <li><b>Multi-Scale:</b> <span style="color: {'#2e7d32' if config.get('training_strategy', {}).get('multi_scale', False) else '#d32f2f'};">{'✅ Enabled' if config.get('training_strategy', {}).get('multi_scale', False) else '❌ Disabled'}</span></li>
            <li><b>Transfer Learning:</b> <span style="color: {'#2e7d32' if model_config.get('transfer_learning', False) else '#d32f2f'};">{'✅ Enabled' if model_config.get('transfer_learning', False) else '❌ Disabled'}</span></li>
            <li><b>Layer Mode:</b> <span style="color: #2e7d32;">{config.get('training_strategy', {}).get('layer_mode', 'single')}</span></li>
        </ul>
    </div>
    """
    
    # Tab 4: Paths
    paths_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #00695c;">📁 Paths & Storage</h5>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
            <li><b>Data Dir:</b> <span style="color: #00695c; font-family: monospace;">{paths_config.get('data_dir', '/data/preprocessed')}</span></li>
            <li><b>Checkpoint Dir:</b> <span style="color: #00695c; font-family: monospace;">{paths_config.get('checkpoint_dir', 'runs/train/checkpoints')}</span></li>
            <li><b>Tensorboard Dir:</b> <span style="color: #00695c; font-family: monospace;">{paths_config.get('tensorboard_dir', 'runs/tensorboard')}</span></li>
        </ul>
    </div>
    """
    
    # Create tabs dengan memastikan semua item adalah widget
    tabs = create_tabs([
        ('Model', widgets.HTML(model_html)),
        ('Hyperparameters', widgets.HTML(hyperparams_html)),
        ('Strategy', widgets.HTML(strategy_html)),
        ('Paths', widgets.HTML(paths_html))
    ])
    
    return tabs

def update_config_tabs(tabs_widget: widgets.Tab, config: Dict[str, Any]) -> widgets.Tab:
    """Update existing tabs dengan config baru"""
    # Buat tab baru dengan config yang diperbarui
    new_tabs = create_config_tabs(config)
    
    # Simpan selected index
    selected_index = tabs_widget.selected_index
    
    # Update children dan titles dengan aman
    tabs_widget.children = new_tabs.children
    
    # Update titles untuk setiap tab
    for i in range(len(tabs_widget.children)):
        tabs_widget.set_title(i, new_tabs.titles[i])
    
    # Kembalikan selected index
    tabs_widget.selected_index = selected_index
    
    return tabs_widget
