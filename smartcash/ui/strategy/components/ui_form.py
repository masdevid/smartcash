"""
File: smartcash/ui/strategy/components/ui_form.py
Deskripsi: Form widgets untuk strategy (non-hyperparameters) dengan one-liner optimizations
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons


def create_strategy_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets khusus strategy parameters"""
    
    # Extract strategy configs dengan safe defaults
    validation = config.get('validation', {})
    utils = config.get('training_utils', {})
    multi_scale = config.get('multi_scale', {})
    model = config.get('model', {})
    
    # Generate dynamic experiment name dengan fallback
    model_type = model.get('model_type', 'efficient_optimized')
    layer_mode = utils.get('layer_mode', 'single')
    default_experiment = f"{model_type}_{layer_mode}"
    
    # Widget creators dengan responsive layout - one-liner factories dengan overflow fix
    int_slider = lambda val, min_val, max_val, desc: widgets.IntSlider(value=val, min=min_val, max=max_val, description=desc, style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    float_slider = lambda val, min_val, max_val, desc, step=0.001: widgets.FloatSlider(value=val, min=min_val, max=max_val, step=step, description=desc, style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'), readout_format='.3f')
    checkbox = lambda val, desc: widgets.Checkbox(value=val, description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width='auto', max_width='100%', overflow='hidden'))
    text_input = lambda val, desc: widgets.Text(value=val, description=desc, style={'description_width': '100px'}, layout=widgets.Layout(width='auto', max_width='100%', overflow='hidden'))
    dropdown = lambda val, opts, desc: widgets.Dropdown(value=val, options=opts, description=desc, style={'description_width': '100px'}, layout=widgets.Layout(width='auto', max_width='100%', overflow='hidden'))
    
    # Strategy-specific form components dengan one-liner mapping
    form_components = {
        # Validation strategy widgets
        'val_frequency_slider': int_slider(validation.get('frequency', 1), 1, 10, 'Val Frequency:'),
        'iou_thres_slider': float_slider(validation.get('iou_thres', 0.6), 0.1, 0.9, 'IoU Threshold:', 0.05),
        'conf_thres_slider': float_slider(validation.get('conf_thres', 0.001), 0.0001, 0.01, 'Conf Threshold:', 0.0001),
        'max_detections_slider': int_slider(validation.get('max_detections', 300), 50, 1000, 'Max Detections:'),
        
        # Training utilities widgets
        'experiment_name_text': text_input(utils.get('experiment_name', default_experiment), 'Experiment:'),
        'checkpoint_dir_text': text_input(utils.get('checkpoint_dir', '/content/runs/train/checkpoints'), 'Checkpoint Dir:'),
        'tensorboard_checkbox': checkbox(utils.get('tensorboard', True), 'TensorBoard'),
        'log_metrics_slider': int_slider(utils.get('log_metrics_every', 10), 1, 50, 'Log Every:'),
        'visualize_batch_slider': int_slider(utils.get('visualize_batch_every', 100), 10, 500, 'Visualize Every:'),
        'gradient_clipping_slider': float_slider(utils.get('gradient_clipping', 1.0), 0.1, 5.0, 'Grad Clipping:', 0.1),
        'layer_mode_dropdown': dropdown(utils.get('layer_mode', 'single'), ['single', 'multilayer'], 'Layer Mode:'),
        
        # Multi-scale training widgets
        'multi_scale_checkbox': checkbox(multi_scale.get('enabled', True), 'Multi-scale'),
        'img_size_min_slider': int_slider(multi_scale.get('img_size_min', 320), 256, 512, 'Min Size:'),
        'img_size_max_slider': int_slider(multi_scale.get('img_size_max', 640), 512, 1024, 'Max Size:')
    }
    
    # Add save/reset buttons dan status panel
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi strategi training",
        reset_tooltip="Reset ke nilai default"
    )
    
    # Merge components dengan save/reset buttons
    form_components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'button_container': save_reset_buttons['container'],
        'status_panel': widgets.Output(layout=widgets.Layout(width='100%', min_height='60px'))
    })
    
    return form_components

def create_config_summary_card(config: Dict[str, Any], last_saved: Optional[str] = None) -> widgets.HTML:
    """Create comprehensive summary card dengan timestamp support - FIXED signature"""
    # Extract all config sections dengan safe defaults
    training = config.get('training', {})
    scheduler = config.get('scheduler', {})
    validation = config.get('validation', {})
    loss = config.get('loss', {})
    utils = config.get('training_utils', {})
    multi_scale = config.get('multi_scale', {})
    early_stopping = config.get('early_stopping', {})
    save_best = config.get('save_best', {}) 

    # Timestamp display dengan conditional formatting
    timestamp_display = f" | 📅 {last_saved}" if last_saved else ""

    # One-liner untuk safe value extraction dengan fallback
    get_val = lambda section, key, default: section.get(key, default)
    bool_icon = lambda val: '✅' if val else '❌'

    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 8px; padding: 16px; margin: 10px 0; color: white;">
        <h4 style="margin: 0 0 12px 0; color: white;">📊 Ringkasan Lengkap Konfigurasi Training{timestamp_display}</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;">
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>🏋️ Training Hyperparameters:</strong><br>
                Epochs: {get_val(training, 'epochs', 100)}<br>
                Batch Size: {get_val(training, 'batch_size', 16)}<br>
                Learning Rate: {get_val(training, 'lr', 0.01)}<br>
                Momentum: {get_val(training, 'momentum', 0.937)}<br>
                Weight Decay: {get_val(training, 'weight_decay', 0.0005)}<br>
                Mixed Precision: {bool_icon(get_val(training, 'mixed_precision', True))}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>📈 Scheduler & Early Stop:</strong><br>
                Type: {get_val(scheduler, 'type', 'cosine').title()}<br>
                Warmup Epochs: {get_val(scheduler, 'warmup_epochs', 3)}<br>
                Patience: {get_val(early_stopping, 'patience', 15)}<br>
                Min Delta: {get_val(early_stopping, 'min_delta', 0.001)}<br>
                Save Best: {bool_icon(get_val(save_best, 'enabled', True))} 
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>✅ Validation Strategy:</strong><br>
                Frequency: {get_val(validation, 'frequency', 1)}<br>
                IoU Threshold: {get_val(validation, 'iou_thres', 0.6)}<br>
                Conf Threshold: {get_val(validation, 'conf_thres', 0.001)}<br>
                Max Detections: {get_val(validation, 'max_detections', 300)}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>📊 Loss Parameters:</strong><br>
                Box Loss: {get_val(loss, 'box_loss_gain', 0.05)}<br>
                Class Loss: {get_val(loss, 'cls_loss_gain', 0.5)}<br>
                Object Loss: {get_val(loss, 'obj_loss_gain', 1.0)}<br>
                Focal Gamma: {get_val(loss, 'fl_gamma', 0.0)}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>🔧 Training Utilities:</strong><br>
                Experiment: {get_val(utils, 'experiment_name', 'efficientnet_b4_training')}<br>
                TensorBoard: {bool_icon(get_val(utils, 'tensorboard', True))}<br>
                Layer Mode: {get_val(utils, 'layer_mode', 'single').title()}<br>
                Grad Clipping: {get_val(utils, 'gradient_clipping', 1.0)}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>🔄 Multi-scale & Logging:</strong><br>
                Multi-scale: {bool_icon(get_val(multi_scale, 'enabled', True))}<br>
                Size Range: {get_val(multi_scale, 'img_size_min', 320)}-{get_val(multi_scale, 'img_size_max', 640)}px<br>
                Log Every: {get_val(utils, 'log_metrics_every', 10)} steps<br>
                Visualize Every: {get_val(utils, 'visualize_batch_every', 100)} steps
            </div>
            
        </div>
    </div>
    """
    
    return widgets.HTML(value=summary_html)
    