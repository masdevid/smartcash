
"""
File: smartcash/ui/strategy/handlers/config_updater.py  
Deskripsi: Updater untuk strategy UI dari configuration
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def update_strategy_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update strategy UI dari config 🔄 - aligned dengan strategy_config.yaml"""
    try:
        validation = config.get('validation', {})
        training_utils = config.get('training_utils', {})
        multi_scale = config.get('multi_scale', {})
        
        # Update validation widgets
        ui_components['val_frequency_slider'].value = validation.get('frequency', 1)
        ui_components['iou_thres_slider'].value = validation.get('iou_thres', 0.6)
        ui_components['conf_thres_slider'].value = validation.get('conf_thres', 0.001)
        ui_components['max_detections_slider'].value = validation.get('max_detections', 300)
        
        # Update training utils widgets
        ui_components['experiment_name_text'].value = training_utils.get('experiment_name', 'efficient_optimized_single')
        ui_components['checkpoint_dir_text'].value = training_utils.get('checkpoint_dir', '/content/runs/train/checkpoints')
        ui_components['tensorboard_checkbox'].value = training_utils.get('tensorboard', True)
        ui_components['log_metrics_slider'].value = training_utils.get('log_metrics', 10)
        ui_components['visualize_batch_slider'].value = training_utils.get('visualize_batch', 100)
        ui_components['layer_mode_dropdown'].value = training_utils.get('layer_mode', 'single')
        
        # Update multi-scale widgets
        ui_components['multi_scale_checkbox'].value = multi_scale.get('enabled', True)
        ui_components['img_size_min_slider'].value = multi_scale.get('img_size_min', 320)
        ui_components['img_size_max_slider'].value = multi_scale.get('img_size_max', 640)
        
        logger.info("✅ Strategy UI berhasil di-update dari config")
        
    except Exception as e:
        logger.error(f"❌ Error update strategy UI: {str(e)}")