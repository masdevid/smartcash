"""
File: smartcash/ui/strategy/components/ui_form.py
Deskripsi: Form widgets untuk strategy (non-hyperparameters)
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons


def create_strategy_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets khusus strategy parameters"""
    
    # Extract strategy configs
    validation = config.get('validation', {})
    utils = config.get('training_utils', {})
    multi_scale = config.get('multi_scale', {})
    model = config.get('model', {})
    
    # Generate dynamic experiment name
    model_type = model.get('model_type', 'efficient_optimized')
    layer_mode = utils.get('layer_mode', 'single')
    default_experiment = f"{model_type}_{layer_mode}"
    
    # Widget creators dengan responsive layout
    int_slider = lambda val, min_val, max_val, desc: widgets.IntSlider(value=val, min=min_val, max=max_val, description=desc, style={'description_width': '120px'}, layout=widgets.Layout(width='100%', max_width='100%'))
    float_slider = lambda val, min_val, max_val, desc, step=0.001: widgets.FloatSlider(value=val, min=min_val, max=max_val, step=step, description=desc, style={'description_width': '120px'}, layout=widgets.Layout(width='100%', max_width='100%'), readout_format='.3f')
    checkbox = lambda val, desc: widgets.Checkbox(value=val, description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width='auto'))
    text_input = lambda val, desc: widgets.Text(value=val, description=desc, style={'description_width': '120px'}, layout=widgets.Layout(width='100%', max_width='100%'))
    dropdown = lambda val, opts, desc: widgets.Dropdown(value=val, options=opts, description=desc, style={'description_width': '120px'}, layout=widgets.Layout(width='100%', max_width='100%'))
    
    # Strategy-specific form components
    form_components = {
        # Validation strategy
        'val_frequency_slider': int_slider(validation.get('frequency', 1), 1, 10, 'Val Frequency:'),
        'iou_thres_slider': float_slider(validation.get('iou_thres', 0.6), 0.1, 0.9, 'IoU Threshold:', 0.05),
        'conf_thres_slider': float_slider(validation.get('conf_thres', 0.001), 0.0001, 0.01, 'Conf Threshold:', 0.0001),
        'max_detections_slider': int_slider(validation.get('max_detections', 300), 50, 1000, 'Max Detections:'),
        
        # Training utilities
        'experiment_name_text': text_input(utils.get('experiment_name', default_experiment), 'Experiment:'),
        'checkpoint_dir_text': text_input(utils.get('checkpoint_dir', '/content/runs/train/checkpoints'), 'Checkpoint Dir:'),
        'tensorboard_checkbox': checkbox(utils.get('tensorboard', True), 'TensorBoard'),
        'log_metrics_slider': int_slider(utils.get('log_metrics_every', 10), 1, 50, 'Log Every:'),
        'visualize_batch_slider': int_slider(utils.get('visualize_batch_every', 100), 10, 500, 'Visualize Every:'),
        'gradient_clipping_slider': float_slider(utils.get('gradient_clipping', 1.0), 0.1, 5.0, 'Grad Clipping:', 0.1),
        'layer_mode_dropdown': dropdown(utils.get('layer_mode', 'single'), ['single', 'multilayer'], 'Layer Mode:'),
        
        # Multi-scale training
        'multi_scale_checkbox': checkbox(multi_scale.get('enabled', True), 'Multi-scale'),
        'img_size_min_slider': int_slider(multi_scale.get('img_size_min', 320), 256, 512, 'Min Size:'),
        'img_size_max_slider': int_slider(multi_scale.get('img_size_max', 640), 512, 1024, 'Max Size:')
    }
    
    # Add save/reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi strategi training",
        reset_tooltip="Reset ke nilai default"
    )
    
    form_components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'button_container': save_reset_buttons['container'],
        'status_panel': widgets.Output(layout=widgets.Layout(width='100%', min_height='60px'))
    })
    
    return form_components


def create_config_summary_card(config: Dict[str, Any]) -> widgets.HTML:
    """Create comprehensive summary card dengan hyperparameters dan strategy"""
    # Extract all config sections
    training = config.get('training', {})
    scheduler = config.get('scheduler', {})
    validation = config.get('validation', {})
    loss = config.get('loss', {})
    utils = config.get('training_utils', {})
    multi_scale = config.get('multi_scale', {})
    early_stopping = config.get('early_stopping', {})
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 8px; padding: 16px; margin: 10px 0; color: white;">
        <h4 style="margin: 0 0 12px 0; color: white;">📊 Ringkasan Lengkap Konfigurasi Training</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;">
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>🏋️ Training Hyperparameters:</strong><br>
                Epochs: {training.get('epochs', 100)}<br>
                Batch Size: {training.get('batch_size', 16)}<br>
                Learning Rate: {training.get('lr', 0.01)}<br>
                Momentum: {training.get('momentum', 0.937)}<br>
                Weight Decay: {training.get('weight_decay', 0.0005)}<br>
                Mixed Precision: {'✅' if training.get('mixed_precision', True) else '❌'}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>📈 Scheduler & Early Stop:</strong><br>
                Type: {scheduler.get('type', 'cosine').title()}<br>
                Warmup Epochs: {scheduler.get('warmup_epochs', 3)}<br>
                Patience: {early_stopping.get('patience', 15)}<br>
                Min Delta: {early_stopping.get('min_delta', 0.001)}<br>
                Save Best: {'✅' if training.get('save_best_only', True) else '❌'}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>✅ Validation Strategy:</strong><br>
                Frequency: {validation.get('frequency', 1)}<br>
                IoU Threshold: {validation.get('iou_thres', 0.6)}<br>
                Conf Threshold: {validation.get('conf_thres', 0.001)}<br>
                Max Detections: {validation.get('max_detections', 300)}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>📊 Loss Parameters:</strong><br>
                Box Loss: {loss.get('box_loss_gain', 0.05)}<br>
                Class Loss: {loss.get('cls_loss_gain', 0.5)}<br>
                Object Loss: {loss.get('obj_loss_gain', 1.0)}<br>
                Focal Gamma: {loss.get('fl_gamma', 0.0)}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>🔧 Training Utilities:</strong><br>
                Experiment: {utils.get('experiment_name', 'efficientnet_b4_training')}<br>
                TensorBoard: {'✅' if utils.get('tensorboard', True) else '❌'}<br>
                Layer Mode: {utils.get('layer_mode', 'single').title()}<br>
                Grad Clipping: {utils.get('gradient_clipping', 1.0)}
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px;">
                <strong>🔄 Multi-scale & Logging:</strong><br>
                Multi-scale: {'✅' if multi_scale.get('enabled', True) else '❌'}<br>
                Size Range: {multi_scale.get('img_size_min', 320)}-{multi_scale.get('img_size_max', 640)}px<br>
                Log Every: {utils.get('log_metrics_every', 10)} steps<br>
                Visualize Every: {utils.get('visualize_batch_every', 100)} steps
            </div>
            
        </div>
    </div>
    """
    
    return widgets.HTML(value=summary_html)