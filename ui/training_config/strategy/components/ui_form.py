"""
File: smartcash/ui/training_config/strategy/components/strategy_form.py
Deskripsi: Form widgets untuk config cell training strategy yang DRY
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons


def create_strategy_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets dengan one-liner style dan config values"""
    ts_config = config.get('strategy', {})
    
    # One-liner helper functions
    slider = lambda val, min_val, max_val, desc, step=None: widgets.FloatSlider(value=val, min=min_val, max=max_val, description=desc, step=step or 0.001, style={'description_width': '150px'}, layout=widgets.Layout(width='100%'))
    int_slider = lambda val, min_val, max_val, desc, step=1: widgets.IntSlider(value=val, min=min_val, max=max_val, description=desc, step=step, style={'description_width': '150px'}, layout=widgets.Layout(width='100%'))
    checkbox = lambda val, desc: widgets.Checkbox(value=val, description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width='100%'))
    text_input = lambda val, desc: widgets.Text(value=val, description=desc, style={'description_width': '150px'}, layout=widgets.Layout(width='100%'))
    dropdown = lambda val, opts, desc: widgets.Dropdown(value=val, options=opts, description=desc, style={'description_width': '150px'}, layout=widgets.Layout(width='100%'))
    radio = lambda val, opts, desc: widgets.RadioButtons(value=val, options=opts, description=desc, style={'description_width': '150px'}, layout=widgets.Layout(width='100%'))
    
    # Extract nested configs dengan defaults
    optimizer = ts_config.get('optimizer', {})
    scheduler = ts_config.get('scheduler', {})
    early_stopping = ts_config.get('early_stopping', {})
    checkpoint = ts_config.get('checkpoint', {})
    utils = ts_config.get('utils', {})
    validation = ts_config.get('validation', {})
    multiscale = ts_config.get('multiscale', {})
    
    # Create form components dengan one-liner style
    form_components = {
        # Parameter utama
        'enabled_checkbox': checkbox(ts_config.get('enabled', True), 'Enable Training Strategy'),
        'batch_size_slider': int_slider(ts_config.get('batch_size', 16), 1, 64, 'Batch Size:'),
        'epochs_slider': int_slider(ts_config.get('epochs', 100), 1, 300, 'Epochs:'),
        'learning_rate_slider': slider(ts_config.get('learning_rate', 0.001), 0.0001, 0.1, 'Learning Rate:'),
        
        # Optimizer
        'optimizer_dropdown': dropdown(optimizer.get('type', 'adam'), ['adam', 'sgd', 'adamw'], 'Optimizer:'),
        'weight_decay_slider': slider(optimizer.get('weight_decay', 0.0005), 0.0, 0.01, 'Weight Decay:', 0.0001),
        'momentum_slider': slider(optimizer.get('momentum', 0.9), 0.0, 1.0, 'Momentum:', 0.1),
        
        # Scheduler
        'scheduler_checkbox': checkbox(scheduler.get('enabled', True), 'Enable Scheduler'),
        'scheduler_dropdown': dropdown(scheduler.get('type', 'cosine'), ['cosine', 'step', 'exponential'], 'Scheduler Type:'),
        'warmup_epochs_slider': int_slider(scheduler.get('warmup_epochs', 5), 0, 20, 'Warmup Epochs:'),
        'min_lr_slider': slider(scheduler.get('min_lr', 0.00001), 0.000001, 0.001, 'Min LR:', 0.000001),
        
        # Early Stopping
        'early_stopping_checkbox': checkbox(early_stopping.get('enabled', True), 'Enable Early Stopping'),
        'patience_slider': int_slider(early_stopping.get('patience', 10), 1, 50, 'Patience:'),
        'min_delta_slider': slider(early_stopping.get('min_delta', 0.001), 0.0001, 0.01, 'Min Delta:', 0.0001),
        
        # Checkpoint
        'checkpoint_checkbox': checkbox(checkpoint.get('enabled', True), 'Enable Checkpoint'),
        'save_best_only_checkbox': checkbox(checkpoint.get('save_best_only', True), 'Save Best Only'),
        'save_freq_slider': int_slider(checkpoint.get('save_freq', 1), 1, 10, 'Save Frequency:'),
        
        # Utils
        'experiment_name': text_input(utils.get('experiment_name', 'efficientnet_b4_training'), 'Experiment Name:'),
        'checkpoint_dir': text_input(utils.get('checkpoint_dir', '/content/runs/train/checkpoints'), 'Checkpoint Dir:'),
        'tensorboard': checkbox(utils.get('tensorboard', True), 'Enable TensorBoard'),
        'log_metrics_every': int_slider(utils.get('log_metrics_every', 10), 1, 50, 'Log Metrics Every:'),
        'visualize_batch_every': int_slider(utils.get('visualize_batch_every', 100), 10, 500, 'Visualize Every:', 10),
        'gradient_clipping': slider(utils.get('gradient_clipping', 1.0), 0.1, 5.0, 'Gradient Clipping:', 0.1),
        'mixed_precision': checkbox(utils.get('mixed_precision', True), 'Enable Mixed Precision'),
        'layer_mode': radio(utils.get('layer_mode', 'single'), ['single', 'multilayer'], 'Layer Mode:'),
        
        # Validation
        'validation_frequency': int_slider(validation.get('validation_frequency', 1), 1, 10, 'Validation Freq:'),
        'iou_threshold': slider(validation.get('iou_threshold', 0.6), 0.1, 0.9, 'IoU Threshold:', 0.05),
        'conf_threshold': slider(validation.get('conf_threshold', 0.001), 0.0001, 0.01, 'Conf Threshold:', 0.0001),
        
        # Multiscale
        'multi_scale': checkbox(multiscale.get('enabled', True), 'Enable Multi-scale Training')
    }
    
    # Add save/reset buttons dengan reusable component
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi strategi pelatihan",
        reset_tooltip="Reset ke nilai default",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive."
    )
    
    # Merge all components
    form_components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'button_container': save_reset_buttons['container'],
        'sync_info': save_reset_buttons['sync_info'],
        'status_panel': widgets.HTML()  # Required untuk ConfigCellInitializer
    })
    
    return form_components