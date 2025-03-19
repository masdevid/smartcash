"""
File: smartcash/ui/dataset/split_config_utils.py
Deskripsi: Utilitas untuk pengaturan konfigurasi split dataset
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_dataset_config(
    dataset_config_path: str = 'configs/dataset_config.yaml',
    default_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load konfigurasi split dataset dari file.
    
    Args:
        dataset_config_path: Path ke file konfigurasi dataset
        default_config: Konfigurasi default jika file tidak ditemukan
        
    Returns:
        Dictionary berisi konfigurasi split dataset
    """
    config = default_config or {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup'
        }
    }
    
    try:
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                
            # Update dengan config yang dimuat
            if 'data' in loaded_config:
                for key, value in loaded_config['data'].items():
                    if key not in config['data']:
                        config['data'][key] = value
                    elif key == 'split_ratios' and isinstance(value, dict):
                        config['data']['split_ratios'].update(value)
                    else:
                        config['data'][key] = value
    except Exception:
        # Gunakan default jika terjadi error
        pass
    
    return config

def save_dataset_config(
    config: Dict[str, Any],
    dataset_config_path: str = 'configs/dataset_config.yaml'
) -> bool:
    """
    Simpan konfigurasi dataset ke file.
    
    Args:
        config: Konfigurasi yang akan disimpan
        dataset_config_path: Path file tujuan
        
    Returns:
        Boolean menandakan keberhasilan operasi
    """
    try:
        # Ekstrak data yang diperlukan untuk dataset_config
        if 'data' not in config:
            return False
            
        dataset_config = {
            'data': {
                key: config['data'][key] for key in [
                    'split_ratios', 'stratified_split', 'random_seed', 
                    'backup_before_split', 'backup_dir'
                ] if key in config['data']
            }
        }
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(dataset_config_path), exist_ok=True)
        
        # Simpan ke file
        with open(dataset_config_path, 'w') as f:
            yaml.dump(dataset_config, f)
            
        return True
    except Exception:
        return False

def get_default_split_config() -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    return {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup'
        }
    }

def normalize_split_percentages(
    train_pct: float, 
    valid_pct: float, 
    test_pct: float
) -> Dict[str, float]:
    """
    Normalisasi persentase split agar totalnya tepat 100%.
    
    Args:
        train_pct: Persentase training
        valid_pct: Persentase validation
        test_pct: Persentase testing
        
    Returns:
        Dictionary berisi nilai rasio yang dinormalisasi
    """
    total = train_pct + valid_pct + test_pct
    
    if abs(total - 100.0) > 0.001:
        # Normalisasi ke 100%
        factor = 100.0 / total
        train_pct *= factor
        valid_pct *= factor
        test_pct *= factor
    
    return {
        'train': train_pct / 100.0,
        'valid': valid_pct / 100.0,
        'test': test_pct / 100.0
    }