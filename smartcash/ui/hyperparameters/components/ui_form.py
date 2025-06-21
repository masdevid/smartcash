"""
File: smartcash/ui/hyperparameters/components/ui_form.py
Deskripsi: Form components untuk hyperparameters dengan parameter essentials saja yang digunakan backend
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.hyperparameters.handlers.defaults import (
    get_optimizer_options, get_scheduler_options, get_checkpoint_metric_options
)
from smartcash.ui.hyperparameters.utils.form_helpers import (
    create_slider_widget, create_int_slider_widget, create_dropdown_widget, 
    create_checkbox_widget, create_summary_cards_widget
)
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.status_panel import create_status_panel


def create_hyperparameters_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat form hyperparameters dengan parameter essentials backend saja 🎯"""
    
    # Extract config dengan fallbacks
    training = config.get('training', {})
    optimizer = config.get('optimizer', {})
    scheduler = config.get('scheduler', {})
    loss = config.get('loss', {})
    early_stopping = config.get('early_stopping', {})
    checkpoint = config.get('checkpoint', {})
    
    # Core training parameters yang digunakan backend
    training_widgets = {
        'epochs_slider': create_int_slider_widget(training.get('epochs', 100), 10, 300, 'Epochs:'),
        'batch_size_slider': create_int_slider_widget(training.get('batch_size', 16), 4, 64, 'Batch Size:', 4),
        'learning_rate_slider': create_slider_widget(training.get('learning_rate', 0.01), 0.0001, 0.1, 0.001, 'Learning Rate:', '.4f'),
        'image_size_slider': create_int_slider_widget(training.get('image_size', 640), 320, 1280, 'Image Size:', 32),
    }
    
    # Optimizer essentials - hanya yang digunakan backend
    optimizer_widgets = {
        'optimizer_dropdown': create_dropdown_widget(optimizer.get('type', 'SGD'), get_optimizer_options(), 'Optimizer:'),
        'weight_decay_slider': create_slider_widget(optimizer.get('weight_decay', 0.0005), 0.0, 0.01, 0.0001, 'Weight Decay:', '.4f'),
    }
    
    # Scheduler essentials
    scheduler_widgets = {
        'scheduler_dropdown': create_dropdown_widget(scheduler.get('type', 'cosine'), get_scheduler_options(), 'Scheduler:'),
        'warmup_epochs_slider': create_int_slider_widget(scheduler.get('warmup_epochs', 3), 0, 10, 'Warmup Epochs:'),
    }
    
    # Loss weights yang digunakan backend
    loss_widgets = {
        'box_loss_gain_slider': create_slider_widget(loss.get('box_loss_gain', 0.05), 0.01, 1.0, 0.01, 'Box Loss Gain:', '.2f'),
        'cls_loss_gain_slider': create_slider_widget(loss.get('cls_loss_gain', 0.5), 0.1, 2.0, 0.1, 'Class Loss Gain:', '.1f'),
        'obj_loss_gain_slider': create_slider_widget(loss.get('obj_loss_gain', 1.0), 0.1, 2.0, 0.1, 'Object Loss Gain:', '.1f'),
    }
    
    # Early stopping dan checkpoint essentials
    control_widgets = {
        'early_stopping_checkbox': create_checkbox_widget(early_stopping.get('enabled', True), 'Enable Early Stopping'),
        'patience_slider': create_int_slider_widget(early_stopping.get('patience', 15), 5, 50, 'Patience (epochs):'),
        'save_best_checkbox': create_checkbox_widget(checkpoint.get('save_best', True), 'Save Best Model'),
        'checkpoint_metric_dropdown': create_dropdown_widget(checkpoint.get('metric', 'mAP_0.5'), get_checkpoint_metric_options(), 'Best Model Metric:'),
    }
    
    # Create summary cards widget untuk overview
    summary_cards = create_summary_cards_widget({
        'Training': f"Epochs: {training.get('epochs', 100)}, Batch: {training.get('batch_size', 16)}, LR: {training.get('learning_rate', 0.01):.4f}",
        'Optimizer': f"{optimizer.get('type', 'SGD')} (decay: {optimizer.get('weight_decay', 0.0005):.4f})",
        'Loss': f"Box: {loss.get('box_loss_gain', 0.05):.2f}, Cls: {loss.get('cls_loss_gain', 0.5):.1f}, Obj: {loss.get('obj_loss_gain', 1.0):.1f}",
        'Control': f"Early Stop: {'On' if early_stopping.get('enabled', True) else 'Off'}, Save Best: {'On' if checkpoint.get('save_best', True) else 'Off'}"
    })
    
    # Action buttons
    save_reset_buttons = create_save_reset_buttons()
    status_panel = create_status_panel()
    
    # Combine all widgets dengan one-liner spread
    return {
        **training_widgets, **optimizer_widgets, **scheduler_widgets, 
        **loss_widgets, **control_widgets,
        'summary_cards': summary_cards,
        'save_reset_buttons': save_reset_buttons,
        'status_panel': status_panel
    }


def _create_section_header(title: str, emoji: str = "📊") -> widgets.HTML:
    """Helper untuk create section header dengan emoji"""
    return widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 8px 12px; border-radius: 6px; 
                    font-weight: bold; margin-bottom: 8px;'>
            {emoji} {title}
        </div>
    """)


# Removed parameters yang tidak digunakan backend:
# - mixed_precision_checkbox (tidak digunakan di backend model)
# - gradient_accumulation_slider (tidak di backend config)
# - gradient_clipping_slider (overlap dengan strategy form)
# - momentum_slider (SGD default, tidak perlu exposed)
# - min_delta_slider (early stopping default sudah cukup)