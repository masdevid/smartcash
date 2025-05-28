"""
File: smartcash/ui/training_config/hyperparameters/handlers/defaults.py
Deskripsi: Default config values untuk hyperparameter dengan structure yang clean
"""

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Return default hyperparameters config dengan one-liner structure"""
    
    return {
        "hyperparameters": {
            "training": {"batch_size": 16, "image_size": 640, "epochs": 100, "dropout": 0.0},
            "optimizer": {"type": "SGD", "learning_rate": 0.01, "weight_decay": 0.0005, "momentum": 0.937},
            "scheduler": {"enabled": True, "type": "cosine", "warmup_epochs": 3},
            "loss": {"box_loss_gain": 0.05, "cls_loss_gain": 0.5, "obj_loss_gain": 1.0},
            "early_stopping": {"enabled": True, "patience": 10, "min_delta": 0.001},
            "augmentation": {"enabled": True},
            "checkpoint": {"save_best": True, "metric": "mAP_0.5"}
        }
    }