"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler untuk operasi konfigurasi split dataset
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import os
from smartcash.ui.utils.constants import ICONS

def load_default_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    # Nilai default untuk split dataset dengan ratio 70-15-15
    return {
        'data': {
            'split': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15,
                'stratified': True
            },
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'dataset_path': 'data',
            'preprocessed_path': 'data/preprocessed'
        }
    }

def load_config() -> Dict[str, Any]:
    """
    Load konfigurasi split dataset dari file YAML atau ConfigManager.
    Returns:
        Dictionary berisi konfigurasi split dataset
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        config_manager = get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
        config = config_manager.get_module_config('dataset_split')
        if config:
            return config
    except Exception as e:
        print(f"{ICONS['warning']} Error saat mengakses ConfigManager: {str(e)}")
    # Fallback ke file
    config_path = Path("configs/dataset_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return load_default_config()

def save_config(config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi split dataset ke file YAML atau ConfigManager.
    Args:
        config: Konfigurasi yang akan disimpan
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        config_manager = get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
        return config_manager.save_module_config('dataset_split', config)
    except Exception as e:
        print(f"{ICONS['warning']} Error saat menyimpan dengan ConfigManager: {str(e)}")
    # Fallback ke file
    config_path = Path("configs/dataset_config.yaml")
    os.makedirs(config_path.parent, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return True

def get_config_manager_instance():
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        return get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
    except Exception as e:
        return None

def save_config_with_manager(config: Dict[str, Any], ui_components: Dict[str, Any], logger=None) -> bool:
    config_manager = get_config_manager_instance()
    if config_manager:
        try:
            config_manager.register_ui_components('dataset_split', ui_components)
            return config_manager.save_module_config('dataset_split', config)
        except Exception as e:
            if logger: logger.warning(f"Gagal menyimpan dengan ConfigManager: {str(e)}")
    return bool(save_config(config))

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Pastikan struktur konfigurasi benar
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    if 'split' not in config['data']:
        config['data']['split'] = {}
    
    # Update nilai split dari slider
    if 'train_slider' in ui_components:
        config['data']['split']['train'] = round(ui_components['train_slider'].value, 2)
    if 'val_slider' in ui_components:
        config['data']['split']['val'] = round(ui_components['val_slider'].value, 2)
    if 'test_slider' in ui_components:
        config['data']['split']['test'] = round(ui_components['test_slider'].value, 2)
    
    # Update nilai stratified dari checkbox
    if 'stratified_checkbox' in ui_components:
        config['data']['split']['stratified'] = ui_components['stratified_checkbox'].value
    
    # Update nilai random seed dari input
    if 'random_seed' in ui_components:
        config['data']['random_seed'] = ui_components['random_seed'].value
    
    # Update nilai backup dari checkbox
    if 'backup_checkbox' in ui_components:
        config['data']['backup_before_split'] = ui_components['backup_checkbox'].value
    
    # Update nilai backup dir dari input
    if 'backup_dir' in ui_components:
        config['data']['backup_dir'] = ui_components['backup_dir'].value
    
    # Update nilai dataset path dari input
    if 'dataset_path' in ui_components:
        config['data']['dataset_path'] = ui_components['dataset_path'].value
    
    # Update nilai preprocessed path dari input
    if 'preprocessed_path' in ui_components:
        config['data']['preprocessed_path'] = ui_components['preprocessed_path'].value
    
    return config
