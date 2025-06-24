
"""
File: smartcash/ui/strategy/handlers/defaults.py
Deskripsi: Default configuration untuk strategy module
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def get_default_strategy_config() -> Dict[str, Any]:
    """Get default strategy configuration 🎯 - aligned dengan strategy_config.yaml"""
    return {
        'validation': {
            'frequency': 1,
            'iou_thres': 0.6,
            'conf_thres': 0.001,
            'max_detections': 300
        },
        'training_utils': {
            'experiment_name': 'efficient_optimized_single',
            'checkpoint_dir': '/content/runs/train/checkpoints',
            'tensorboard': True,
            'log_metrics': 10,
            'visualize_batch': 100,
            'layer_mode': 'single'
        },
        'multi_scale': {
            'enabled': True,
            'img_size_min': 320,
            'img_size_max': 640
        }
    }