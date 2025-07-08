"""
File: smartcash/ui/model/evaluate/configs/evaluation_defaults.py
Description: Default configuration values for evaluation module
"""

# Default evaluation configuration
EVALUATION_DEFAULTS = {
    "scenarios": {
        "position_variation": {
            "enabled": True,
            "augmentation_config": {
                "num_variations": 5,
                "rotation_range": [-30, 30],
                "translation_range": [-0.2, 0.2],
                "scale_range": [0.8, 1.2]
            }
        },
        "lighting_variation": {
            "enabled": True,
            "augmentation_config": {
                "num_variations": 5,
                "brightness_range": [-0.3, 0.3],
                "contrast_range": [0.7, 1.3],
                "gamma_range": [0.7, 1.3]
            }
        }
    },
    "models": {
        "cspdarknet": {
            "enabled": True,
            "checkpoint_pattern": "checkpoints/cspdarknet/*.pt"
        },
        "efficientnet_b4": {
            "enabled": True, 
            "checkpoint_pattern": "checkpoints/efficientnet_b4/*.pt"
        }
    },
    "inference": {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 100,
        "img_size": 640,
        "half": True,
        "fuse": True,
        "nms": True
    },
    "evaluation": {
        "batch_size": 16,
        "num_workers": 4,
        "device": "auto",
        "save_results": True,
        "save_visualizations": True
    }
}