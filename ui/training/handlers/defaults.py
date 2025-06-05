"""
File: smartcash/ui/training/handlers/defaults.py
Deskripsi: Konfigurasi default dan utilitas YAML untuk training dengan struktur yang jelas dan one-liner style
"""

from typing import Dict, Any, List, Optional, Callable
import os
import yaml
from smartcash.common.config.manager import get_config_manager

def get_default_training_config() -> Dict[str, Any]:
    """Mendapatkan konfigurasi default untuk training dengan struktur yang jelas"""
    # Konfigurasi default dengan struktur yang jelas menggunakan one-liner style
    return {
        'model': {
            'type': 'efficient_basic',
            'backbone': 'efficientnet_b4',
            'confidence': 0.25,
            'iou_threshold': 0.45
        },
        'training': {
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 0.01,
            'optimizer': 'SGD',
            'scheduler': 'cosine',
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'nesterov': True
        },
        'augmentation': {
            'enabled': True,
            'types': ['geometric', 'color', 'noise'],
            'intensity': 0.5
        },
        'validation': {
            'iou_thres': 0.6,
            'conf_thres': 0.001,
            'frequency': 1
        },
        'dataset': {
            'train_path': '/content/dataset/train',
            'val_path': '/content/dataset/val',
            'classes': ['1000', '2000', '5000', '10000', '20000', '50000', '100000']
        },
        'training_utils': {
            'experiment_name': 'efficientnet_b4_training',
            'checkpoint_dir': '/content/runs/train/checkpoints',
            'tensorboard': True,
            'log_metrics_every': 10,
            'visualize_batch_every': 100,
            'gradient_clipping': 1.0,
            'mixed_precision': True
        }
    }

def load_yaml_config(config_name: str = 'training') -> Dict[str, Any]:
    """Load konfigurasi dari ConfigManager dengan error handling yang baik"""
    try:
        # Get config manager dengan one-liner
        config_manager = get_config_manager()
        
        # Load config dengan one-liner
        return config_manager.get_config(config_name) or {}
        
    except Exception as e:
        print(f"⚠️ Error loading config '{config_name}': {str(e)}")
        return {}

def save_yaml_config(config: Dict[str, Any], config_name: str = 'training') -> bool:
    """Simpan konfigurasi ke ConfigManager dengan error handling yang baik"""
    try:
        # Get config manager dengan one-liner
        config_manager = get_config_manager()
        
        # Save config dengan one-liner
        config_manager.set_config(config_name, config)
        return True
        
    except Exception as e:
        print(f"⚠️ Error saving config '{config_name}': {str(e)}")
        return False

def get_merged_config(config_names: List[str] = None) -> Dict[str, Any]:
    """Mendapatkan konfigurasi gabungan dari beberapa sumber dengan one-liner style"""
    # Default config names jika tidak ada yang diberikan
    config_names = config_names or ['model', 'training', 'hyperparameters', 'augmentation']
    
    # Get config manager dengan one-liner
    config_manager = get_config_manager()
    
    # Load dan merge semua configs dengan one-liner
    merged_config = {}
    [merged_config.update(config_manager.get_config(name) or {}) for name in config_names]
    
    # Jika merged config kosong, gunakan default
    return merged_config if merged_config else get_default_training_config()

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], ui_map: Dict[str, tuple] = None) -> None:
    """Update UI components dari config dengan one-liner style"""
    # Default UI map jika tidak ada yang diberikan
    ui_map = ui_map or {
        'model_type': ('type', None, 'model'),
        'backbone_selector': ('backbone', None, 'model'),
        'confidence_threshold': ('confidence', None, 'model'),
        'iou_threshold': ('iou_threshold', None, 'model'),
        'batch_size': ('batch_size', None, 'training'),
        'epochs': ('epochs', None, 'training'),
        'learning_rate': ('learning_rate', None, 'training'),
        'optimizer': ('optimizer', None, 'training')
    }
    
    # Update semua UI components dengan one-liner
    [setattr(ui_components.get(ui_key), 'value', config.get(item[2], {}).get(item[0], item[1])) 
     for ui_key, item in ui_map.items() 
     if ui_key in ui_components and hasattr(ui_components.get(ui_key), 'value')]

def extract_config_from_ui(ui_components: Dict[str, Any], extraction_map: Dict[str, tuple] = None) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner style"""
    # Default extraction map jika tidak ada yang diberikan
    extraction_map = extraction_map or {
        'model': ['model_type', 'backbone_selector', 'confidence_threshold', 'iou_threshold'],
        'training': ['batch_size', 'epochs', 'learning_rate', 'optimizer']
    }
    
    # Extract config dengan nested dict comprehension
    return {section: {comp.replace(f"{section}_", ""): ui_components.get(comp).value 
                     for comp in comps if comp in ui_components} 
            for section, comps in extraction_map.items()}

def register_config_callbacks(ui_components: Dict[str, Any], callback: Callable[[Dict[str, Any]], None]) -> None:
    """Register callbacks untuk UI components dengan one-liner style"""
    # Register callbacks untuk semua UI components dengan list comprehension
    [ui_components[comp].observe(lambda change: callback(extract_config_from_ui(ui_components)), names='value')
     for comp in ui_components 
     if hasattr(ui_components.get(comp), 'observe') 
     and comp not in ['main_container', 'container', 'ui', 'logger']]
