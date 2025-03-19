"""
File: smartcash/ui/dataset/split_config_utils.py
Deskripsi: Utilitas untuk pengaturan konfigurasi split dataset
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def load_dataset_config(config_path: str = 'configs/dataset_config.yaml') -> Dict[str, Any]:
    """Load konfigurasi split dataset dari file."""
    default_config = {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup'
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                
            # Update dengan config yang dimuat
            if 'data' in loaded_config:
                for key, value in loaded_config['data'].items():
                    if key == 'split_ratios' and isinstance(value, dict):
                        default_config['data']['split_ratios'].update(value)
                    else:
                        default_config['data'][key] = value
    except Exception:
        pass
    
    return default_config

def save_dataset_config(config: Dict[str, Any], config_path: str = 'configs/dataset_config.yaml') -> bool:
    """Simpan konfigurasi dataset ke file."""
    try:
        # Buat config baru untuk disimpan (hindari reference)
        dataset_config = {'data': {}}
        
        # Salin konfigurasi yang diperlukan
        for key in ['split_ratios', 'stratified_split', 'random_seed', 'backup_before_split', 'backup_dir']:
            if key in config['data']:
                dataset_config['data'][key] = config['data'][key]
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Simpan ke file
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        return True
    except Exception as e:
        print(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False

def normalize_split_percentages(train_pct: float, valid_pct: float, test_pct: float) -> Dict[str, float]:
    """Normalisasi persentase split agar totalnya tepat 100%."""
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

def extract_config_values(ui_components: Dict[str, Any]) -> Tuple[float, float, float, bool, int, bool, str]:
    """Ekstrak nilai konfigurasi dari komponen UI."""
    train_pct = ui_components['split_sliders'][0].value
    val_pct = ui_components['split_sliders'][1].value
    test_pct = ui_components['split_sliders'][2].value
    stratified = ui_components['stratified'].value
    random_seed = ui_components['advanced_options'].children[0].value
    backup_before_split = ui_components['advanced_options'].children[1].value
    backup_dir = ui_components['advanced_options'].children[2].value
    
    return train_pct, val_pct, test_pct, stratified, random_seed, backup_before_split, backup_dir

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Update konfigurasi dari komponen UI."""
    train_pct, val_pct, test_pct, stratified, random_seed, backup_before_split, backup_dir = extract_config_values(ui_components)
    
    # Normalize percentages
    split_ratios = normalize_split_percentages(train_pct, val_pct, test_pct)
    
    # Update config
    config['data']['split_ratios'] = split_ratios
    config['data']['stratified_split'] = stratified
    config['data']['random_seed'] = random_seed
    config['data']['backup_before_split'] = backup_before_split
    config['data']['backup_dir'] = backup_dir
    
    return config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update komponen UI dari konfigurasi."""
    # Split sliders
    split_ratios = config['data'].get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
    ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
    ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
    ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
    
    # Stratified checkbox
    ui_components['stratified'].value = config['data'].get('stratified_split', True)
    
    # Advanced options
    ui_components['advanced_options'].children[0].value = config['data'].get('random_seed', 42)
    ui_components['advanced_options'].children[1].value = config['data'].get('backup_before_split', True)
    ui_components['advanced_options'].children[2].value = config['data'].get('backup_dir', 'data/splits_backup')