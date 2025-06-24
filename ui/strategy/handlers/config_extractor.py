"""
File: smartcash/ui/strategy/handlers/config_extractor.py
Deskripsi: Extractor untuk strategy configuration dari UI components
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def extract_strategy_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract strategy config dari UI components 🎯"""
    try:
        config = {
            'validation': {
                'frequency': ui_components['val_frequency_slider'].value,
                'iou_thres': ui_components['iou_thres_slider'].value,
                'conf_thres': ui_components['conf_thres_slider'].value,
                'max_detections': ui_components['max_detections_slider'].value
            },
            'training_utils': {
                'experiment_name': ui_components['experiment_name_text'].value,
                'checkpoint_dir': ui_components['checkpoint_dir_text'].value,
                'tensorboard': ui_components['tensorboard_checkbox'].value,
                'log_metrics': ui_components['log_metrics_slider'].value,
                'visualize_batch': ui_components['visualize_batch_slider'].value,
                'layer_mode': ui_components['layer_mode_dropdown'].value
            },
            'multi_scale': {
                'enabled': ui_components['multi_scale_checkbox'].value,
                'img_size_min': ui_components['img_size_min_slider'].value,
                'img_size_max': ui_components['img_size_max_slider'].value
            }
        }
        
        logger.info("✅ Strategy config berhasil di-extract")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error extract strategy config: {str(e)}")
        return {}