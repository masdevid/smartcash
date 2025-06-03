"""
File: smartcash/ui/training_config/training_strategy/handlers/ui_updater.py
Deskripsi: Update UI dari config dengan one-liner style yang DRY sesuai struktur YAML
"""

from typing import Dict, Any


def update_training_strategy_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan one-liner style sesuai struktur training_config.yaml"""
    
    # One-liner safe setter untuk memastikan keamanan akses UI components
    set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Extract nested configs dengan fallbacks
    validation = config.get('validation', {})
    multi_scale = config.get('multi_scale', True)  # Multi-scale adalah boolean di YAML
    training_utils = config.get('training_utils', {})
    
    # Update komponen validation UI dengan field yang sesuai YAML
    set_val('validation_frequency_slider', validation.get('frequency', 1))
    set_val('iou_threshold_slider', validation.get('iou_thres', 0.6))  # Field name dari YAML
    set_val('conf_threshold_slider', validation.get('conf_thres', 0.001))  # Field name dari YAML
    
    # Update komponen multi-scale UI (hanya boolean di YAML)
    set_val('multi_scale_checkbox', multi_scale)  # Langsung gunakan nilai boolean
    
    # Update komponen training utils UI sesuai field di YAML
    set_val('experiment_name_text', training_utils.get('experiment_name', 'efficientnet_b4_training'))
    set_val('checkpoint_dir_text', training_utils.get('checkpoint_dir', '/content/runs/train/checkpoints'))
    set_val('tensorboard_checkbox', training_utils.get('tensorboard', True))
    set_val('log_metrics_every_slider', training_utils.get('log_metrics_every', 10))
    set_val('visualize_batch_every_slider', training_utils.get('visualize_batch_every', 100))
    set_val('gradient_clipping_slider', training_utils.get('gradient_clipping', 1.0))
    set_val('mixed_precision_checkbox', training_utils.get('mixed_precision', True))
    set_val('layer_mode_dropdown', training_utils.get('layer_mode', 'single'))
    
    # Logging untuk informasi
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], f"📊 UI training strategy berhasil diperbarui", "info")
    # Parameter lainnya sudah diatur di atas, tidak perlu diulang atau ditambahkan 
    # yang tidak ada di struktur YAML