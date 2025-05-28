"""
File: smartcash/ui/training_config/hyperparameters/components/main_components.py
Deskripsi: Form components untuk hyperparameters dengan reusable widgets
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.utils.constants import ICONS


def create_hyperparameters_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat form hyperparameters dengan reusable components"""
    
    # One-liner widget creation dari config
    hp_config = config.get('hyperparameters', {})
    training = hp_config.get('training', {})
    optimizer = hp_config.get('optimizer', {})
    scheduler = hp_config.get('scheduler', {})
    early_stopping = hp_config.get('early_stopping', {})
    augmentation = hp_config.get('augmentation', {})
    checkpoint = hp_config.get('checkpoint', {})
    
    # Basic parameters dengan one-liner creation
    form_widgets = {
        'batch_size_slider': widgets.IntSlider(value=training.get('batch_size', 16), min=1, max=128, description='Batch Size:', style={'description_width': '120px'}),
        'image_size_slider': widgets.IntSlider(value=training.get('image_size', 640), min=320, max=1280, step=32, description='Image Size:', style={'description_width': '120px'}),
        'epochs_slider': widgets.IntSlider(value=training.get('epochs', 100), min=1, max=500, description='Epochs:', style={'description_width': '120px'}),
        'dropout_slider': widgets.FloatSlider(value=training.get('dropout', 0.0), min=0.0, max=0.5, step=0.01, description='Dropout:', readout_format='.2f', style={'description_width': '120px'}),
        
        # Optimizer parameters
        'optimizer_dropdown': widgets.Dropdown(options=['SGD', 'Adam', 'AdamW', 'RMSprop'], value=optimizer.get('type', 'SGD'), description='Optimizer:', style={'description_width': '120px'}),
        'learning_rate_slider': widgets.FloatLogSlider(value=optimizer.get('learning_rate', 0.01), base=10, min=-5, max=-1, description='Learning Rate:', readout_format='.6f', style={'description_width': '120px'}),
        'weight_decay_slider': widgets.FloatLogSlider(value=optimizer.get('weight_decay', 0.0005), base=10, min=-6, max=-2, description='Weight Decay:', readout_format='.6f', style={'description_width': '120px'}),
        'momentum_slider': widgets.FloatSlider(value=optimizer.get('momentum', 0.937), min=0.8, max=0.999, step=0.001, description='Momentum:', readout_format='.3f', style={'description_width': '120px'}),
        
        # Scheduler parameters
        'scheduler_checkbox': widgets.Checkbox(value=scheduler.get('enabled', True), description='Gunakan Scheduler'),
        'scheduler_dropdown': widgets.Dropdown(options=['cosine', 'linear', 'step', 'exp', 'none'], value=scheduler.get('type', 'cosine'), description='LR Scheduler:', style={'description_width': '120px'}),
        'warmup_epochs_slider': widgets.IntSlider(value=scheduler.get('warmup_epochs', 3), min=0, max=10, description='Warmup Epochs:', style={'description_width': '120px'}),
        
        # Loss parameters
        'box_loss_gain_slider': widgets.FloatSlider(value=hp_config.get('loss', {}).get('box_loss_gain', 0.05), min=0.01, max=0.1, step=0.01, description='Box Loss Gain:', readout_format='.2f', style={'description_width': '120px'}),
        'cls_loss_gain_slider': widgets.FloatSlider(value=hp_config.get('loss', {}).get('cls_loss_gain', 0.5), min=0.1, max=1.0, step=0.1, description='Class Loss Gain:', readout_format='.1f', style={'description_width': '120px'}),
        'obj_loss_gain_slider': widgets.FloatSlider(value=hp_config.get('loss', {}).get('obj_loss_gain', 1.0), min=0.5, max=2.0, step=0.1, description='Object Loss Gain:', readout_format='.1f', style={'description_width': '120px'}),
        
        # Early stopping parameters
        'early_stopping_checkbox': widgets.Checkbox(value=early_stopping.get('enabled', True), description='Early Stopping'),
        'patience_slider': widgets.IntSlider(value=early_stopping.get('patience', 10), min=1, max=50, description='Patience:', style={'description_width': '120px'}),
        'min_delta_slider': widgets.FloatSlider(value=early_stopping.get('min_delta', 0.001), min=0.0001, max=0.01, step=0.0001, description='Min Delta:', readout_format='.4f', style={'description_width': '120px'}),
        
        # Augmentation and checkpoint
        'augment_checkbox': widgets.Checkbox(value=augmentation.get('enabled', True), description='Gunakan Augmentasi'),
        'save_best_checkbox': widgets.Checkbox(value=checkpoint.get('save_best', True), description='Simpan Model Terbaik'),
        'checkpoint_metric_dropdown': widgets.Dropdown(options=['mAP_0.5', 'mAP_0.5:0.95', 'precision', 'recall', 'f1', 'loss'], value=checkpoint.get('metric', 'mAP_0.5'), description='Metrik Checkpoint:', style={'description_width': '120px'})
    }
    
    # Grouped widgets dengan responsive layout
    basic_group = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('settings', '⚙️')} Parameter Dasar</h4>"),
        form_widgets['batch_size_slider'],
        form_widgets['image_size_slider'],
        form_widgets['epochs_slider'],
        form_widgets['dropout_slider']
    ], layout=widgets.Layout(width='100%', padding='10px', border='1px solid #ddd', border_radius='5px'))
    
    optimization_group = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('optimization', '🔄')} Optimasi</h4>"),
        form_widgets['optimizer_dropdown'],
        form_widgets['learning_rate_slider'],
        form_widgets['weight_decay_slider'],
        form_widgets['momentum_slider'],
        widgets.HTML("<hr style='margin: 10px 0'>"),
        form_widgets['scheduler_checkbox'],
        form_widgets['scheduler_dropdown'],
        form_widgets['warmup_epochs_slider']
    ], layout=widgets.Layout(width='100%', padding='10px', border='1px solid #ddd', border_radius='5px'))
    
    advanced_group = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('advanced', '🔧')} Parameter Lanjutan</h4>"),
        form_widgets['augment_checkbox'],
        widgets.HTML("<b>Parameter Loss</b>"),
        form_widgets['box_loss_gain_slider'],
        form_widgets['cls_loss_gain_slider'],
        form_widgets['obj_loss_gain_slider'],
        widgets.HTML("<hr style='margin: 10px 0'>"),
        widgets.HTML("<b>Early Stopping</b>"),
        form_widgets['early_stopping_checkbox'],
        form_widgets['patience_slider'],
        form_widgets['min_delta_slider'],
        widgets.HTML("<hr style='margin: 10px 0'>"),
        widgets.HTML("<b>Checkpoint</b>"),
        form_widgets['save_best_checkbox'],
        form_widgets['checkpoint_metric_dropdown']
    ], layout=widgets.Layout(width='100%', padding='10px', border='1px solid #ddd', border_radius='5px'))
    
    # Save/reset buttons menggunakan shared component
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi hyperparameter",
        reset_tooltip="Reset ke nilai default",
        with_sync_info=True,
        sync_message="Konfigurasi akan disinkronkan dengan Google Drive."
    )
    
    # Status panel menggunakan shared component
    status_panel = create_status_panel()
    
    # Three-column layout dengan responsive design
    form_layout = widgets.VBox([
        create_header("Konfigurasi Hyperparameter", "Pengaturan parameter untuk training model", ICONS.get('settings', '⚙️')),
        status_panel,
        widgets.HBox([basic_group, optimization_group, advanced_group], layout=widgets.Layout(width='100%', justify_content='space-between', align_items='stretch', gap='10px')),
        save_reset_buttons['container'],
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Return components dengan naming yang konsisten
    return {
        'main_container': form_layout,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'status_panel': status_panel,
        **form_widgets
    }