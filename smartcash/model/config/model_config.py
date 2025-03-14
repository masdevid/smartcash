"""
File: smartcash/model/config/model_config.py
Deskripsi: Konfigurasi dasar model SmartCash dengan dukungan load/save YAML
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

class ModelConfig:
    """
    Konfigurasi dasar model SmartCash dengan dukungan load/save YAML
    dan validasi parameter konfigurasi.
    """
    
    # Konfigurasi default untuk model
    DEFAULT_CONFIG = {
        'model': {
            'name': 'smartcash_model',
            'img_size': [640, 640],
            'batch_size': 16,
            'workers': 4,
            'backbone': 'efficientnet_b4'
        },
        'training': {
            'epochs': 100,
            'lr': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'fliplr': 0.5,
            'flipud': 0.0,
            'scale': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'translate': 0.1,
            'degrees': 0.0
        },
        'layers': ['banknote'],
        'data': {
            'train': 'data/train',
            'val': 'data/valid',
            'test': 'data/test',
            'preprocessing': {
                'cache_dir': '.cache/smartcash'
            }
        },
        'optimizer': {
            'type': 'SGD',
            'params': {}
        },
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'params': {}
        },
        'checkpoint': {
            'save_dir': 'runs/train/weights',
            'save_interval': 10,
            'save_last': True,
            'save_best': True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Inisialisasi konfigurasi model.
        
        Args:
            config_path: Path ke file konfigurasi YAML (opsional)
            **kwargs: Parameter konfigurasi tambahan
        """
        # Muat config default sebagai basis
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Jika ada file konfigurasi, muat dari file
        if config_path:
            config_file = self.load_from_yaml(config_path)
            self._deep_update(self.config, config_file)
        
        # Update dengan parameter dari kwargs
        if kwargs:
            self._update_from_kwargs(kwargs)
        
        # Validasi konfigurasi
        self._validate_config()
    
    def _update_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Update konfigurasi dari keyword arguments.
        
        Args:
            kwargs: Dictionary parameter konfigurasi
        """
        # Update parameter konfigurasi secara langsung tanpa hierarki
        flat_updates = {}
        for key, value in kwargs.items():
            if '.' in key:
                # Format hierarki seperti 'model.img_size'
                parts = key.split('.')
                current = self.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Non-hierarki seperti 'epochs'
                flat_updates[key] = value
        
        # Update flat parameters yang umum digunakan
        common_mappings = {
            'epochs': 'training.epochs',
            'lr': 'training.lr',
            'batch_size': 'model.batch_size',
            'img_size': 'model.img_size',
            'workers': 'model.workers'
        }
        
        for key, value in flat_updates.items():
            if key in common_mappings:
                parts = common_mappings[key].split('.')
                current = self.config
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            elif key in self.config:
                self.config[key] = value
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Update dictionary secara rekursif.
        
        Args:
            target: Dictionary yang akan diupdate
            source: Dictionary sumber update
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self) -> None:
        """Validasi konfigurasi untuk memastikan parameter yang diperlukan tersedia."""
        # Minimal validasi untuk model.img_size
        if 'model' not in self.config or 'img_size' not in self.config['model']:
            raise ValueError("❌ Konfigurasi tidak valid: 'model.img_size' diperlukan")
        
        img_size = self.config['model']['img_size']
        if not isinstance(img_size, list) or len(img_size) != 2:
            raise ValueError("❌ Konfigurasi tidak valid: 'model.img_size' harus berupa list [width, height]")
    
    def load_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """
        Muat konfigurasi dari file YAML.
        
        Args:
            yaml_path: Path ke file YAML
            
        Returns:
            Dictionary konfigurasi
        """
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            raise ValueError(f"❌ Gagal memuat konfigurasi dari {yaml_path}: {str(e)}")
    
    def save_to_yaml(self, yaml_path: str) -> None:
        """
        Simpan konfigurasi ke file YAML.
        
        Args:
            yaml_path: Path untuk menyimpan file YAML
        """
        try:
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            
            # Simpan ke file
            with open(yaml_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            raise ValueError(f"❌ Gagal menyimpan konfigurasi ke {yaml_path}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dapatkan nilai konfigurasi dengan support untuk format dot notation.
        
        Args:
            key: Key konfigurasi (format 'section.key' atau 'key')
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default
        """
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts:
                if part not in current:
                    return default
                current = current[part]
            return current
        else:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi dengan support untuk format dot notation.
        
        Args:
            key: Key konfigurasi (format 'section.key' atau 'key')
            value: Nilai yang akan di-set
        """
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update konfigurasi dengan dictionary baru.
        
        Args:
            updates: Dictionary dengan nilai konfigurasi baru
        """
        self._deep_update(self.config, updates)
    
    def __getitem__(self, key: str) -> Any:
        """Akses nilai konfigurasi dengan operator [] dan dukungan dot notation."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set nilai konfigurasi dengan operator [] dan dukungan dot notation."""
        self.set(key, value)