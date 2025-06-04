#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: /Users/masdevid/Projects/smartcash/smartcash/tests/test_config_compatibility.py
# Deskripsi: Test untuk memastikan kompatibilitas konfigurasi setelah refaktor

import os
import sys
import yaml
import unittest
from pathlib import Path
from typing import Dict, Any, List

# Tambahkan root project ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestConfigCompatibility(unittest.TestCase):
    """Test untuk memastikan kompatibilitas konfigurasi setelah refaktor."""
    
    def setUp(self):
        """Setup untuk test."""
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.config_files = [
            "base_config.yaml",
            "augmentation_config.yaml",
            "colab_config.yaml",
            "dataset_config.yaml",
            "evaluation_config.yaml",
            "hyperparameters_config.yaml",
            "model_config.yaml",
            "preprocessing_config.yaml",
            "training_config.yaml"
        ]
        
        # Struktur kunci yang diharapkan dari setiap file konfigurasi
        self.expected_keys = {
            "augmentation_config.yaml": [
                "augmentation", "cleanup", "performance"
            ],
            "colab_config.yaml": [
                "drive", "environment", "model", "performance", "ui"
            ],
            "dataset_config.yaml": [
                "dataset", "cache", "cleanup"
            ],
            "evaluation_config.yaml": [
                "evaluation", "model", "performance", "scenarios"
            ],
            "hyperparameters_config.yaml": [
                "training", "scheduler", "regularization", "loss", "anchor", 
                "early_stopping", "save_best"
            ],
            "model_config.yaml": [
                "model", "efficientnet", "transfer_learning", "regularization",
                "export", "experiments"
            ],
            "preprocessing_config.yaml": [
                "preprocessing", "augmentation_reference", "cleanup", "performance"
            ],
            "training_config.yaml": [
                "validation", "multi_scale", "training_utils"
            ]
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load konfigurasi dari file YAML."""
        config_path = self.config_dir / config_file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _resolve_base_configs(self, config: Dict[str, Any], config_file: str) -> Dict[str, Any]:
        """Resolve base config inheritance."""
        if '_base_' not in config:
            return config
        
        base_configs = config['_base_']
        if not isinstance(base_configs, list):
            base_configs = [base_configs]
        
        # Load dan merge base configs
        merged_config = {}
        for base_config in base_configs:
            base_config_path = self.config_dir / base_config
            with open(base_config_path, 'r', encoding='utf-8') as f:
                base_config_data = yaml.safe_load(f)
                # Rekursif resolve base config
                base_config_data = self._resolve_base_configs(base_config_data, base_config)
                # Merge dengan config saat ini
                self._deep_merge(merged_config, base_config_data)
        
        # Hapus _base_ dan merge dengan config saat ini
        del config['_base_']
        self._deep_merge(merged_config, config)
        
        return merged_config
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge dua dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _check_keys_exist(self, config: Dict[str, Any], expected_keys: List[str], path: str = "") -> List[str]:
        """Check apakah semua kunci yang diharapkan ada dalam config."""
        missing_keys = []
        for key in expected_keys:
            if "." in key:
                # Handle nested keys
                parts = key.split(".", 1)
                if parts[0] not in config:
                    missing_keys.append(f"{path}.{key}" if path else key)
                elif isinstance(config[parts[0]], dict):
                    nested_missing = self._check_keys_exist(config[parts[0]], [parts[1]], f"{path}.{parts[0]}" if path else parts[0])
                    missing_keys.extend(nested_missing)
            elif key not in config:
                missing_keys.append(f"{path}.{key}" if path else key)
        return missing_keys
    
    def test_config_files_exist(self):
        """Test apakah semua file konfigurasi ada."""
        for config_file in self.config_files:
            config_path = self.config_dir / config_file
            self.assertTrue(config_path.exists(), f"File konfigurasi {config_file} tidak ditemukan")
    
    def test_config_inheritance(self):
        """Test apakah inheritance konfigurasi bekerja dengan benar."""
        for config_file in self.config_files:
            if config_file == "base_config.yaml":
                continue
            
            config = self._load_config(config_file)
            self.assertIn('_base_', config, f"File {config_file} tidak memiliki _base_ config")
    
    def test_resolved_configs_have_expected_keys(self):
        """Test apakah konfigurasi yang di-resolve memiliki semua kunci yang diharapkan."""
        for config_file, expected_keys in self.expected_keys.items():
            config = self._load_config(config_file)
            resolved_config = self._resolve_base_configs(config, config_file)
            
            # Check apakah semua kunci yang diharapkan ada
            missing_keys = self._check_keys_exist(resolved_config, expected_keys)
            self.assertEqual(len(missing_keys), 0, 
                            f"File {config_file} tidak memiliki kunci yang diharapkan: {missing_keys}")
    
    def test_config_structure_compatibility(self):
        """Test untuk memastikan struktur konfigurasi kompatibel dengan yang asli."""
        for config_file in self.config_files:
            if config_file == "base_config.yaml":
                continue
                
            config = self._load_config(config_file)
            resolved_config = self._resolve_base_configs(config, config_file)
            
            # Check apakah struktur konfigurasi sesuai dengan yang diharapkan
            if config_file in self.expected_keys:
                for key in self.expected_keys[config_file]:
                    self.assertIn(key, resolved_config, 
                                f"Kunci {key} tidak ditemukan dalam {config_file} setelah resolve")


if __name__ == "__main__":
    unittest.main()
