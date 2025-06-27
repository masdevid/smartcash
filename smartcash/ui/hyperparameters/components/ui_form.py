"""
File: smartcash/ui/hyperparameters/components/ui_form.py
Deskripsi: Form components untuk hyperparameters dengan parameter penting dan widgets yang clean
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
    """Buat form hyperparameters dengan parameter penting saja dan layout yang compact"""
    
    # Extract config dengan fallbacks
    training = config.get('training', {})
    optimizer = config.get('optimizer', {})
    scheduler = config.get('scheduler', {})
    loss = config.get('loss', {})
    early_stopping = config.get('early_stopping', {})
    checkpoint = config.get('checkpoint', {})
    
    # Training parameter widgets dengan one-liner creation
    training_widgets = {
        'epochs_slider': create_int_slider_widget(training.get('epochs', 100), 10, 300, 'Epochs:'),
        'batch_size_slider': create_int_slider_widget(training.get('batch_size', 16), 4, 64, 'Batch Size:', 4),
        'learning_rate_slider': create_slider_widget(training.get('learning_rate', 0.01), 0.0001, 0.1, 0.001, 'Learning Rate:', '.4f'),
        'image_size_slider': create_int_slider_widget(training.get('image_size', 640), 320, 1280, 'Image Size:', 32),
        'mixed_precision_checkbox': create_checkbox_widget(training.get('mixed_precision', True), 'Mixed Precision'),
        'gradient_accumulation_slider': create_int_slider_widget(training.get('gradient_accumulation', 1), 1, 8, 'Grad Accumulation:'),
        'gradient_clipping_slider': create_slider_widget(training.get('gradient_clipping', 1.0), 0.1, 5.0, 0.1, 'Grad Clipping:', '.1f')
    }
    
    # Optimizer parameter widgets
    optimizer_widgets = {
        'optimizer_dropdown': create_dropdown_widget(optimizer.get('type', 'SGD'), get_optimizer_options(), 'Optimizer:'),
        'weight_decay_slider': create_slider_widget(optimizer.get('weight_decay', 0.0005), 0.0, 0.01, 0.0001, 'Weight Decay:', '.4f'),
        'momentum_slider': create_slider_widget(optimizer.get('momentum', 0.937), 0.8, 0.999, 0.001, 'Momentum:', '.3f')
    }
    
    # Scheduler parameter widgets
    scheduler_widgets = {
        'scheduler_dropdown': create_dropdown_widget(scheduler.get('type', 'cosine'), get_scheduler_options(), 'Scheduler:'),
        'warmup_epochs_slider': create_int_slider_widget(scheduler.get('warmup_epochs', 3), 0, 10, 'Warmup Epochs:')
    }
    
    # Loss parameter widgets
    loss_widgets = {
        'box_loss_gain_slider': create_slider_widget(loss.get('box_loss_gain', 0.05), 0.01, 0.2, 0.01, 'Box Loss:', '.2f'),
        'cls_loss_gain_slider': create_slider_widget(loss.get('cls_loss_gain', 0.5), 0.1, 2.0, 0.1, 'Class Loss:', '.1f'),
        'obj_loss_gain_slider': create_slider_widget(loss.get('obj_loss_gain', 1.0), 0.1, 3.0, 0.1, 'Object Loss:', '.1f')
    }
    
    # Early stopping dan checkpoint widgets
    control_widgets = {
        'early_stopping_checkbox': create_checkbox_widget(early_stopping.get('enabled', True), 'Early Stopping'),
        'patience_slider': create_int_slider_widget(early_stopping.get('patience', 15), 1, 50, 'Patience:'),
        'min_delta_slider': create_slider_widget(early_stopping.get('min_delta', 0.001), 0.0001, 0.01, 0.0001, 'Min Delta:', '.4f'),
        'save_best_checkbox': create_checkbox_widget(checkpoint.get('save_best', True), 'Save Best Model'),
        'checkpoint_metric_dropdown': create_dropdown_widget(checkpoint.get('metric', 'mAP_0.5'), get_checkpoint_metric_options(), 'Best Metric:')
    }
    
    # Summary cards untuk menampilkan config yang tersimpan
    summary_cards = create_summary_cards_widget()
    
    # Save/reset buttons tanpa sync info
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi hyperparameter",
        reset_tooltip="Reset ke nilai default",
        with_sync_info=False
    )
    
    status_panel = create_status_panel("ℹ️ Siap untuk konfigurasi hyperparameter", "info")
    
    # Combine all widgets untuk return
    all_widgets = {
        **training_widgets,
        **optimizer_widgets, 
        **scheduler_widgets,
        **loss_widgets,
        **control_widgets,
        'summary_cards': summary_cards,
        'status_panel': status_panel,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'button_container': save_reset_buttons['container']
    }
    
    return all_widgets