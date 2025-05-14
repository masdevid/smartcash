"""
File: smartcash/common/default_config.py
Deskripsi: Utilitas untuk membuat konfigurasi default jika belum ada file konfigurasi
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE, DEFAULT_PREPROCESSED_DIR, DEFAULT_AUGMENTED_DIR, DEFAULT_VISUALIZATION_DIR

def generate_default_config() -> Dict[str, Any]:
    """
    Buat konfigurasi default SmartCash.
    
    Returns:
        Dictionary berisi konfigurasi dasar
    """
    return {
        "project": {
            "name": "SmartCash",
            "version": "1.0.0",
            "description": "Sistem deteksi denominasi mata uang dengan YOLOv5 dan EfficientNet"
        },
        "drive": {
            "use_drive": True,
            "sync_on_start": True,
            "sync_strategy": "drive_priority",
            "symlinks": True,
            "paths": {
                "smartcash_dir": "SmartCash",
                "configs_dir": "configs",
                "data_dir": "data",
                "runs_dir": "runs",
                "logs_dir": "logs"
            }
        },
        "data": {
            "dir": "data",
            "preprocessed_dir": DEFAULT_PREPROCESSED_DIR,
            "split_ratios": {
                "train": 0.7,
                "valid": 0.15,
                "test": 0.15
            },
            "stratified_split": True,
            "random_seed": 42,
            "source": "roboflow",
            "roboflow": {
                "api_key": "",
                "workspace": "smartcash-wo2us",
                "project": "rupiah-emisi-2022",
                "version": "3"
            }
        },
        "layers": [
            "banknote",
            "nominal",
            "security"
        ],
        "model": {
            "backbone": "efficientnet_b4",
            "input_size": DEFAULT_IMG_SIZE,
            "confidence": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 100,
            "transfer_learning": True,
            "pretrained": True,
            "workers": 4,
            "freeze_backbone": False,
            "unfreeze_epoch": 0
        },
        "training": {
            "epochs": 30,
            "batch_size": 16,
            "early_stopping_patience": 10,
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_patience": 3,
            "lr_factor": 0.1,
            "save_best_only": False,
            "save_every": 0
        },
        "augmentation": {
            "enabled": True,
            "num_variations": 2,
            "output_prefix": "aug",
            "process_bboxes": True,
            "output_dir": DEFAULT_AUGMENTED_DIR,
            "validate_results": False,
            "resume": False,
            "types": [
                "position",
                "lighting",
                "combined"
            ],
            "position": {
                "fliplr": 0.5,
                "flipud": 0.0,
                "degrees": 10,
                "translate": 0.1,
                "scale": 0.1,
                "shear": 0.0,
                "rotation_prob": 0.5,
                "max_angle": 10,
                "flip_prob": 0.5,
                "scale_ratio": 0.1
            },
            "lighting": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "contrast": 0.3,
                "brightness": 0.3,
                "compress": 0.0,
                "brightness_prob": 0.5,
                "brightness_limit": 0.3,
                "contrast_prob": 0.5,
                "contrast_limit": 0.3
            },
            "extreme": {
                "rotation_min": 30,
                "rotation_max": 90,
                "probability": 0.3
            },
            "combined": {
                "enabled": True,
                "prob": 0.5
            }
        },
        "preprocessing": {
            "output_dir": DEFAULT_PREPROCESSED_DIR,
            "save_visualizations": True,
            "vis_dir": DEFAULT_VISUALIZATION_DIR,
            "sample_size": 0,
            "validate": {
                "enabled": True,
                "fix_issues": True,
                "move_invalid": True,
                "visualize": False,
                "check_image_quality": True,
                "check_labels": True,
                "check_coordinates": True
            },
            "normalization": {
                "enabled": True,
                "method": "minmax",
                "target_size": DEFAULT_IMG_SIZE,
                "preserve_aspect_ratio": True,
                "normalize_pixel_values": True,
                "pixel_range": [0, 1]
            }
        },
        "logging": {
            "level": "INFO",
            "use_colors": True,
            "use_emojis": True,
            "log_to_file": True,
            "logs_dir": "logs"
        },
        "environment": {
            "colab": True,
            "create_dirs_on_start": True
        }
    }

def ensure_base_config_exists(config_path: str = "configs/base_config.yaml") -> bool:
    """
    Pastikan file konfigurasi dasar ada, buat jika belum ada.
    
    Args:
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menunjukkan apakah file baru dibuat
    """
    config_file = Path(config_path)
    
    # Cek apakah sudah ada
    if config_file.exists():
        return False
    
    # Buat direktori jika belum ada
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Buat konfigurasi default
    default_config = generate_default_config()
    
    # Tulis ke file
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"✅ File konfigurasi dasar dibuat: {config_path}")
    return True

def ensure_colab_config_exists(config_path: str = "configs/colab_config.yaml") -> bool:
    """
    Pastikan file konfigurasi Colab ada, buat jika belum ada.
    
    Args:
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menunjukkan apakah file baru dibuat
    """
    config_file = Path(config_path)
    
    # Cek apakah sudah ada
    if config_file.exists():
        return False
    
    # Buat direktori jika belum ada
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Buat konfigurasi Colab default
    colab_config = {
        "_base_": "base_config.yaml",
        "drive": {
            "use_drive": True,
            "sync_on_start": True,
            "sync_strategy": "drive_priority",
            "symlinks": True
        },
        "environment": {
            "colab": True,
            "create_dirs_on_start": True,
            "required_packages": [
                "albumentations>=1.1.0",
                "roboflow>=0.2.29",
                "PyYAML>=6.0",
                "tqdm>=4.64.0",
                "ultralytics>=8.0.0",
                "pandas>=1.3.5",
                "seaborn>=0.11.2"
            ]
        },
        "model": {
            "use_tpu": False,
            "use_gpu": True,
            "precision": "mixed_float16",
            "batch_size_auto": True,
            "workers": 2
        },
        "performance": {
            "auto_garbage_collect": True,
            "checkpoint_to_drive": True,
            "release_memory": True
        }
    }
    
    # Tulis ke file
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(colab_config, f, default_flow_style=False)
    
    print(f"✅ File konfigurasi Colab dibuat: {config_path}")
    return True

def ensure_all_configs_exist():
    """Pastikan semua file konfigurasi dasar ada."""
    # Pastikan direktori configs ada
    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Buat konfigurasi dasar jika belum ada
    base_created = ensure_base_config_exists()
    colab_created = ensure_colab_config_exists()
    
    return base_created or colab_created