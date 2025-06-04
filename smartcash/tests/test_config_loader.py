#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: /Users/masdevid/Projects/smartcash/smartcash/tests/test_config_loader.py
# Deskripsi: Test untuk memastikan config loader bekerja dengan benar setelah refaktor

import os
import sys
import yaml
import unittest
from pathlib import Path
from typing import Dict, Any, List, Union

# Tambahkan root project ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ConfigLoader:
    """Utilitas untuk memuat dan menggabungkan konfigurasi YAML."""
    
    def __init__(self, config_dir: Union[str, Path]):
        """
        Inisialisasi ConfigLoader.
        
        Args:
            config_dir: Direktori yang berisi file konfigurasi YAML.
        """
        self.config_dir = Path(config_dir)
        self.cache = {}
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Memuat konfigurasi dari file YAML dan menyelesaikan inheritance.
        
        Args:
            config_file: Nama file konfigurasi (relatif terhadap config_dir).
            
        Returns:
            Dict konfigurasi yang sudah di-resolve.
        """
        # Gunakan cache jika sudah pernah dimuat
        if config_file in self.cache:
            return self.cache[config_file]
        
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Muat konfigurasi dari file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Resolve base configs jika ada
        if '_base_' in config:
            base_configs = config['_base_']
            if not isinstance(base_configs, list):
                base_configs = [base_configs]
            
            # Muat dan gabungkan base configs
            merged_config = {}
            for base_config in base_configs:
                base_config_data = self.load_config(base_config)
                self._deep_merge(merged_config, base_config_data)
            
            # Hapus _base_ dan gabungkan dengan config saat ini
            del config['_base_']
            self._deep_merge(merged_config, config)
            config = merged_config
        
        # Simpan ke cache
        self.cache[config_file] = config
        return config
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge dua dictionary.
        
        Args:
            target: Dictionary target yang akan diupdate.
            source: Dictionary sumber yang akan digabungkan ke target.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value


class TestConfigLoader(unittest.TestCase):
    """Test untuk memastikan ConfigLoader bekerja dengan benar."""
    
    def setUp(self):
        """Setup untuk test."""
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.loader = ConfigLoader(self.config_dir)
        
        # Daftar file konfigurasi yang akan diuji
        self.config_files = [
            "augmentation_config.yaml",
            "colab_config.yaml",
            "dataset_config.yaml",
            "evaluation_config.yaml",
            "hyperparameters_config.yaml",
            "model_config.yaml",
            "preprocessing_config.yaml",
            "training_config.yaml"
        ]
        
        # Kunci-kunci penting yang harus ada di setiap konfigurasi setelah resolve
        self.important_keys = {
            "augmentation_config.yaml": {
                "augmentation.num_variations": 3,
                "augmentation.validate_results": True,
                "augmentation.position.rotate_max_deg": 15,
                "augmentation.lighting.brightness_range": [0.8, 1.2],
                "cleanup.augmentation_patterns": ["aug_.*", ".*_augmented.*", ".*_modified.*", ".*_processed.*"],
                "performance.batch_size": 16
            },
            "colab_config.yaml": {
                "drive.use_drive": True,
                "drive.symlinks": True,
                "environment.colab": True,
                "model.workers": 2,
                "ui.max_displayed_items": 10,
                "performance.auto_garbage_collect": True
            },
            "dataset_config.yaml": {
                "data.validation.enabled": True,
                "data.validation.fix_issues": True,
                "dataset.backup.enabled": True,
                "dataset.export.enabled": True,
                "cache.dir": ".cache/smartcash/dataset"
            },
            "evaluation_config.yaml": {
                "evaluation.test_batch_size": 4,
                "evaluation.confidence_thresholds": [0.1, 0.25, 0.5, 0.75, 0.9],
                "evaluation.per_class_metrics": True,
                "evaluation.visualization.max_samples": 50,
                "model.max_detections": 300,
                "scenarios": [
                    {"name": "pencahayaan_ideal", "subset": "test/pencahayaan_ideal", "description": "Kondisi pencahayaan normal dan ideal"},
                    {"name": "pencahayaan_rendah", "subset": "test/pencahayaan_rendah", "description": "Kondisi pencahayaan rendah (<100 lux)"},
                    {"name": "pencahayaan_tinggi", "subset": "test/pencahayaan_tinggi", "description": "Kondisi pencahayaan tinggi (>1000 lux)"},
                    {"name": "posisi_bervariasi", "subset": "test/posisi_bervariasi", "description": "Uang dengan posisi dan orientasi bervariasi"}
                ]
            },
            "hyperparameters_config.yaml": {
                "training.epochs": 100,
                "training.lr": 0.01,
                "scheduler.type": "cosine",
                "scheduler.warmup_epochs": 3,
                "early_stopping.patience": 15,
                "early_stopping.enabled": True,
                "save_best.enabled": True
            },
            "model_config.yaml": {
                "model.type": "efficient_basic",
                "model.use_efficient_blocks": True,
                "model.use_attention": False,
                "model.use_residual": False,
                "model.use_ciou": False,
                "efficientnet.width_coefficient": 1.4,
                "efficientnet.depth_coefficient": 1.8,
                "transfer_learning.unfreeze_after_epochs": 10
            },
            "preprocessing_config.yaml": {
                "preprocessing.vis_dir": "visualizations/preprocessing",
                "preprocessing.sample_size": 500,
                "preprocessing.validate.visualize": True,
                "preprocessing.analysis.enabled": True,
                "augmentation_reference.preprocessing_variations": 3,
                "cleanup.backup_dir": "data/backup/preprocessing",
                "performance.batch_size": 32,
                "performance.compression_level": 90
            },
            "training_config.yaml": {
                "validation.frequency": 1,
                "validation.iou_thres": 0.6,
                "validation.conf_thres": 0.001,
                "multi_scale": True,
                "training_utils.experiment_name": "efficientnet_b4_training",
                "training_utils.checkpoint_dir": "/content/runs/train/checkpoints",
                "training_utils.tensorboard": True,
                "training_utils.mixed_precision": True
            }
        }
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """
        Mendapatkan nilai dari nested dictionary berdasarkan path kunci.
        
        Args:
            config: Dictionary konfigurasi.
            key_path: Path kunci dengan format "key1.key2.key3".
            
        Returns:
            Nilai yang ditemukan atau None jika tidak ditemukan.
        """
        keys = key_path.split(".")
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value
    
    def test_config_loader_loads_all_files(self):
        """Test apakah ConfigLoader dapat memuat semua file konfigurasi."""
        for config_file in self.config_files:
            config = self.loader.load_config(config_file)
            self.assertIsInstance(config, dict, f"Konfigurasi {config_file} tidak berhasil dimuat")
    
    def test_important_keys_exist_with_correct_values(self):
        """Test apakah kunci-kunci penting ada dengan nilai yang benar setelah resolve."""
        for config_file, important_keys in self.important_keys.items():
            config = self.loader.load_config(config_file)
            
            for key_path, expected_value in important_keys.items():
                actual_value = self._get_nested_value(config, key_path)
                self.assertIsNotNone(actual_value, 
                                    f"Kunci {key_path} tidak ditemukan dalam {config_file}")
                self.assertEqual(actual_value, expected_value,
                                f"Nilai untuk {key_path} dalam {config_file} tidak sesuai: "
                                f"expected={expected_value}, actual={actual_value}")
    
    def test_training_config_inherits_correctly(self):
        """Test apakah training_config.yaml mewarisi dengan benar dari hyperparameters_config.yaml dan model_config.yaml."""
        training_config = self.loader.load_config("training_config.yaml")
        
        # Cek nilai dari hyperparameters_config.yaml
        self.assertEqual(training_config["training"]["epochs"], 100,
                        "training_config.yaml tidak mewarisi epochs dari hyperparameters_config.yaml")
        
        # Cek nilai dari model_config.yaml
        self.assertEqual(training_config["model"]["type"], "efficient_basic",
                        "training_config.yaml tidak mewarisi model.type dari model_config.yaml")
    
    def test_inheritance_chain(self):
        """Test rantai pewarisan konfigurasi."""
        # training_config mewarisi dari hyperparameters_config dan model_config
        # hyperparameters_config dan model_config mewarisi dari base_config
        
        # Muat konfigurasi
        base_config = self.loader.load_config("base_config.yaml")
        hyperparameters_config = self.loader.load_config("hyperparameters_config.yaml")
        model_config = self.loader.load_config("model_config.yaml")
        training_config = self.loader.load_config("training_config.yaml")
        
        # Cek nilai yang diwarisi dari base_config ke hyperparameters_config
        self.assertEqual(hyperparameters_config["project"]["name"], base_config["project"]["name"],
                        "hyperparameters_config tidak mewarisi project.name dari base_config")
        
        # Cek nilai yang diwarisi dari base_config ke model_config
        self.assertEqual(model_config["project"]["name"], base_config["project"]["name"],
                        "model_config tidak mewarisi project.name dari base_config")
        
        # Cek nilai yang diwarisi dari hyperparameters_config ke training_config
        self.assertEqual(training_config["training"]["epochs"], hyperparameters_config["training"]["epochs"],
                        "training_config tidak mewarisi training.epochs dari hyperparameters_config")
        
        # Cek nilai yang diwarisi dari model_config ke training_config
        self.assertEqual(training_config["model"]["type"], model_config["model"]["type"],
                        "training_config tidak mewarisi model.type dari model_config")


if __name__ == "__main__":
    unittest.main()
