"""
File: smartcash/ui/training_config/hyperparameters/handlers/default_config.py
Deskripsi: Konfigurasi default untuk hyperparameters model
"""

from typing import Dict, Any

def get_default_hyperparameters_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk hyperparameters.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    return {
        "hyperparameters": {
            "optimizer": {
                "type": "SGD",
                "learning_rate": 0.01,
                "weight_decay": 0.0005,
                "momentum": 0.937,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8
            },
            "scheduler": {
                "enabled": True,
                "type": "cosine",
                "warmup_epochs": 3,
                "min_lr": 1e-5,
                "patience": 10,
                "factor": 0.1,
                "threshold": 0.001
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "alpha": 0.25,
                "gamma": 2.0,
                "label_smoothing": 0.0,
                "box_loss_gain": 0.05,
                "cls_loss_gain": 0.5,
                "obj_loss_gain": 1.0
            },
            "augmentation": {
                "enabled": True,
                "mosaic": True,
                "mixup": False,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic_prob": 1.0,
                "mixup_prob": 1.0
            },
            "training": {
                "batch_size": 16,
                "image_size": 640,
                "epochs": 100,
                "dropout": 0.0
            },
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.001
            },
            "checkpoint": {
                "save_best": True,
                "metric": "mAP_0.5"
            }
        }
    }