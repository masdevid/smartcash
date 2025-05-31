"""
File: smartcash/model/config/model_config.py
Deskripsi: Konfigurasi dasar model SmartCash dengan dukungan load/save YAML
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from smartcash.dataset.utils.dataset_constants import DEFAULT_TRAIN_DIR, DEFAULT_VALID_DIR, DEFAULT_TEST_DIR, DEFAULT_CHECKPOINT_DIR
from smartcash.model.config.model_constants import DEFAULT_MODEL_CONFIG_FULL

class ModelConfig:
    """Konfigurasi model SmartCash dengan dukungan load/save YAML dan validasi parameter."""
    
    # Menggunakan DEFAULT_MODEL_CONFIG_FULL dari model_constants.py
    DEFAULT_CONFIG = DEFAULT_MODEL_CONFIG_FULL.copy()
    
    DEFAULT_CONFIG.update({
        'data': {'train': DEFAULT_TRAIN_DIR, 'val': DEFAULT_VALID_DIR, 'test': DEFAULT_TEST_DIR, 'preprocessing': {'cache_dir': '.cache/smartcash'}},
        'checkpoint': {'save_dir': DEFAULT_CHECKPOINT_DIR, 'save_interval': 10, 'save_last': True, 'save_best': True}
    })
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Inisialisasi konfigurasi model dari file YAML atau kwargs."""
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path: self._deep_update(self.config, self.load_from_yaml(config_path))
        if kwargs: self._update_from_kwargs(kwargs)
        self._validate_config()
    
    def _update_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Update konfigurasi dari keyword arguments."""
        # Proses kunci dengan notasi dot
        flat_updates = {}
        for key, value in kwargs.items():
            if '.' in key:
                # Langsung update nested dictionary
                parts = key.split('.')
                current = self.config
                for part in parts[:-1]:
                    if part not in current: current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else: flat_updates[key] = value
        
        # Mapping kunci umum ke lokasi sebenarnya dalam konfigurasi
        common_mappings = {
            'epochs': 'training.epochs', 'lr': 'training.lr', 'batch_size': 'model.batch_size',
            'img_size': 'model.img_size', 'workers': 'model.workers'
        }
        
        # Update konfigurasi berdasarkan mapping atau langsung jika ada di level atas
        for key, value in flat_updates.items():
            if key in common_mappings:
                parts = common_mappings[key].split('.')
                current = self.config
                for part in parts[:-1]: current = current[part]
                current[parts[-1]] = value
            elif key in self.config: self.config[key] = value
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Update dictionary secara rekursif dari source ke target."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict): self._deep_update(target[key], value)
            else: target[key] = value
    
    def _validate_config(self) -> None:
        """Validasi konfigurasi untuk memastikan parameter yang diperlukan tersedia."""
        if 'model' not in self.config or 'img_size' not in self.config['model']: raise ValueError("❌ Konfigurasi tidak valid: 'model.img_size' diperlukan")
        img_size = self.config['model']['img_size']
        if not isinstance(img_size, list) or len(img_size) != 2: raise ValueError("❌ Konfigurasi tidak valid: 'model.img_size' harus berupa list [width, height]")
    
    def load_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Muat konfigurasi dari file YAML dan kembalikan sebagai dictionary."""
        try:
            with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
            return config or {}
        except Exception as e: raise ValueError(f"❌ Gagal memuat konfigurasi dari {yaml_path}: {str(e)}")
    
    def save_to_yaml(self, yaml_path: str) -> None:
        """Simpan konfigurasi ke file YAML dengan format yang mudah dibaca."""
        try:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as f: yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e: raise ValueError(f"❌ Gagal menyimpan konfigurasi ke {yaml_path}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dapatkan nilai konfigurasi dengan support untuk format dot notation."""
        if '.' not in key: return self.config.get(key, default)
        
        # Navigasi melalui nested dictionary menggunakan dot notation
        parts, current = key.split('.'), self.config
        for part in parts:
            if part not in current: return default
            current = current[part]
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set nilai konfigurasi dengan support untuk format dot notation."""
        if '.' not in key: 
            self.config[key] = value
            return
            
        # Navigasi dan buat nested dictionary jika diperlukan
        parts, current = key.split('.'), self.config
        for part in parts[:-1]:
            if part not in current: current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update konfigurasi dengan dictionary baru secara rekursif."""
        self._deep_update(self.config, updates)
    
    def __getitem__(self, key: str) -> Any: return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None: self.set(key, value)