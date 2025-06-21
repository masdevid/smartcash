# File: smartcash/ui/hyperparameters/components/ui_form.py
# Deskripsi: Form components untuk hyperparameters - menggunakan fallback_utils

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
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


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
        'device_dropdown': create_dropdown_widget(training.get('device', 'auto'), ['auto', 'cuda', 'cpu'], 'Device:')
    }
    
    # Optimizer parameters
    optimizer_widgets = {
        'optimizer_dropdown': create_dropdown_widget(optimizer.get('name', 'AdamW'), get_optimizer_options(), 'Optimizer:'),
        'weight_decay_slider': create_slider_widget(optimizer.get('weight_decay', 0.0001), 0, 0.01, 0.0001, 'Weight Decay:', '.4f'),
        'momentum_slider': create_slider_widget(optimizer.get('momentum', 0.937), 0.5, 0.99, 0.01, 'Momentum:', '.3f')
    }
    
    # Scheduler parameters
    scheduler_widgets = {
        'scheduler_dropdown': create_dropdown_widget(scheduler.get('name', 'CosineAnnealingLR'), get_scheduler_options(), 'Scheduler:'),
        'warmup_epochs_slider': create_int_slider_widget(scheduler.get('warmup_epochs', 3), 0, 20, 'Warmup Epochs:'),
        'min_lr_slider': create_slider_widget(scheduler.get('min_lr', 1e-6), 1e-8, 1e-3, 1e-7, 'Min LR:', '.1e')
    }
    
    # Loss function parameters
    loss_widgets = {
        'box_loss_gain_slider': create_slider_widget(loss.get('box_loss_gain', 0.05), 0.01, 0.2, 0.01, 'Box Loss Gain:', '.3f'),
        'cls_loss_gain_slider': create_slider_widget(loss.get('cls_loss_gain', 0.5), 0.1, 2.0, 0.1, 'Cls Loss Gain:', '.2f'),
        'obj_loss_gain_slider': create_slider_widget(loss.get('obj_loss_gain', 1.0), 0.1, 2.0, 0.1, 'Obj Loss Gain:', '.2f'),
        'focal_loss_checkbox': create_checkbox_widget(loss.get('focal_loss', False), 'Use Focal Loss'),
        'label_smoothing_slider': create_slider_widget(loss.get('label_smoothing', 0.0), 0.0, 0.2, 0.01, 'Label Smoothing:', '.3f')
    }
    
    # Early stopping parameters
    early_stopping_widgets = {
        'early_stopping_checkbox': create_checkbox_widget(early_stopping.get('enabled', True), 'Enable Early Stopping'),
        'patience_slider': create_int_slider_widget(early_stopping.get('patience', 10), 3, 50, 'Patience:'),
        'min_delta_slider': create_slider_widget(early_stopping.get('min_delta', 0.001), 0.0001, 0.01, 0.0001, 'Min Delta:', '.4f'),
        'monitor_dropdown': create_dropdown_widget(early_stopping.get('monitor', 'val_loss'), 
                                                  ['val_loss', 'val_mAP', 'val_precision', 'val_recall'], 'Monitor:')
    }
    
    # Checkpoint parameters
    checkpoint_widgets = {
        'save_best_checkbox': create_checkbox_widget(checkpoint.get('save_best', True), 'Save Best Model'),
        'save_interval_slider': create_int_slider_widget(checkpoint.get('save_interval', 10), 1, 50, 'Save Interval:'),
        'max_checkpoints_slider': create_int_slider_widget(checkpoint.get('max_checkpoints', 5), 1, 20, 'Max Checkpoints:'),
        'metric_dropdown': create_dropdown_widget(checkpoint.get('metric', 'mAP'), get_checkpoint_metric_options(), 'Best Metric:')
    }
    
    # Summary cards
    summary_cards = create_summary_cards_widget(config)
    
    # Status panel dan save/reset buttons
    status_panel = create_status_panel()
    save_reset_buttons = create_save_reset_buttons()
    
    # Gabung semua widgets
    all_widgets = {
        **training_widgets,
        **optimizer_widgets,
        **scheduler_widgets,
        **loss_widgets,
        **early_stopping_widgets,
        **checkpoint_widgets,
        'summary_cards': summary_cards,
        'status_panel': status_panel,
        'save_reset_buttons': save_reset_buttons
    }
    
    logger.info("✅ Hyperparameters form created successfully")
    return all_widgets


def update_summary_cards(widgets_dict: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update summary cards dengan nilai terbaru dari widgets 🔄"""
    try_operation_safe(
        operation=lambda: _update_summary_cards_content(widgets_dict, config),
        fallback_value=None,
        logger=logger,
        operation_name="updating summary cards"
    )


def _update_summary_cards_content(widgets_dict: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Internal logic untuk update summary cards"""
    # Update config dengan nilai dari widgets
    training = config.get('training', {})
    optimizer = config.get('optimizer', {})
    scheduler = config.get('scheduler', {})
    
    widget_mappings = [
        ('epochs_slider', training, 'epochs'),
        ('batch_size_slider', training, 'batch_size'),
        ('learning_rate_slider', training, 'learning_rate'),
        ('optimizer_dropdown', optimizer, 'name'),
        ('weight_decay_slider', optimizer, 'weight_decay'),
        ('momentum_slider', optimizer, 'momentum'),
        ('scheduler_dropdown', scheduler, 'name')
    ]
    
    for widget_key, config_section, param in widget_mappings:
        if widget_key in widgets_dict:
            config_section[param] = widgets_dict[widget_key].value
    
    # Update summary cards
    if 'summary_cards' in widgets_dict:
        updated_config = {'training': training, 'optimizer': optimizer, 'scheduler': scheduler}
        widgets_dict['summary_cards'].value = create_summary_cards_widget(updated_config).value