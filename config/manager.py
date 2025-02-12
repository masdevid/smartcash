# File: src/config/manager.py
# Author: Alfrida Sabar
# Deskripsi: Sistem manajemen konfigurasi terpusat untuk SmartCash Detector

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    img_size: int = 640
    nc: int = 7
    backbone_phi: int = 4
    anchors: list = None
    conf_thres: float = 0.25
    
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stop_patience: int = 5
    
@dataclass
class DataConfig:
    data_dir: Path = Path('data/rupiah')
    train_path: Path = None
    val_path: Path = None
    test_path: Path = None
    class_names: list = None
    aug_factor: int = 2

class ConfigManager:
    def __init__(self, config_path: str = 'config/default.yaml'):
        self.config_path = Path(config_path)
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.data = DataConfig()
        self._load_config()
        
    def _load_config(self) -> None:
        """Load konfigurasi dari YAML"""
        if not self.config_path.exists():
            self._save_default_config()
            return
            
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
            
        # Update configs
        for k, v in cfg.get('model', {}).items():
            setattr(self.model, k, v)
        for k, v in cfg.get('train', {}).items():
            setattr(self.train, k, v)
        for k, v in cfg.get('data', {}).items():
            if k.endswith('_path'):
                v = Path(v)
            setattr(self.data, k, v)
            
    def _save_default_config(self) -> None:
        """Simpan konfigurasi default"""
        config = {
            'model': {
                'img_size': self.model.img_size,
                'nc': self.model.nc,
                'backbone_phi': self.model.backbone_phi,
                'anchors': [[10,13], [16,30], [33,23],
                           [30,61], [62,45], [59,119],
                           [116,90], [156,198], [373,326]],
                'conf_thres': self.model.conf_thres
            },
            'train': {
                'epochs': self.train.epochs,
                'batch_size': self.train.batch_size,
                'learning_rate': self.train.learning_rate,
                'weight_decay': self.train.weight_decay,
                'early_stop_patience': self.train.early_stop_patience
            },
            'data': {
                'data_dir': str(self.data.data_dir),
                'train_path': 'data/rupiah/train',
                'val_path': 'data/rupiah/val', 
                'test_path': 'data/rupiah/test',
                'class_names': ['100k', '10k', '1k', '20k', '2k', '50k', '5k'],
                'aug_factor': self.data.aug_factor
            }
        }
        
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)