"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler untuk operasi konfigurasi split dataset
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import os
from smartcash.ui.utils.constants import ICONS
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.io import load_config, save_config

logger = get_logger(__name__)

def get_default_split_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
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

def get_split_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi split dataset dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary berisi konfigurasi split dataset
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_module_config('dataset_split')
        
        if config:
            return config
            
        # Jika tidak ada config, gunakan default
        logger.warning("⚠️ Konfigurasi split dataset tidak ditemukan, menggunakan default")
        return get_default_split_config()
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi split dataset: {str(e)}")
        return get_default_split_config()

def update_config_from_ui(ui_components: Dict[str, Any], config_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi split dataset dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        config_manager: Config manager instance (opsional)
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get config manager if not provided
        if config_manager is None:
            config_manager = get_config_manager()
        
        # Get current config
        config = config_manager.get_module_config('dataset_split') or get_default_split_config()
        
        # Update config from UI
        if 'train_slider' in ui_components:
            config['data']['split']['train'] = ui_components['train_slider'].value
            
        if 'val_slider' in ui_components:
            config['data']['split']['val'] = ui_components['val_slider'].value
            
        if 'test_slider' in ui_components:
            config['data']['split']['test'] = ui_components['test_slider'].value
            
        if 'stratified_checkbox' in ui_components:
            config['data']['split']['stratified'] = ui_components['stratified_checkbox'].value
            
        if 'random_seed_slider' in ui_components:
            config['data']['random_seed'] = ui_components['random_seed_slider'].value
            
        if 'backup_checkbox' in ui_components:
            config['data']['backup_before_split'] = ui_components['backup_checkbox'].value
            
        if 'backup_dir' in ui_components:
            config['data']['backup_dir'] = ui_components['backup_dir'].value
            
        # Save config
        config_manager.save_module_config('dataset_split', config)
        
        logger.info("✅ Konfigurasi split dataset berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi split dataset: {str(e)}")
        return get_default_split_config()

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            config = get_split_config(ui_components)
            
        # Update UI components
        if 'train_slider' in ui_components:
            ui_components['train_slider'].value = config['data']['split']['train']
            
        if 'val_slider' in ui_components:
            ui_components['val_slider'].value = config['data']['split']['val']
            
        if 'test_slider' in ui_components:
            ui_components['test_slider'].value = config['data']['split']['test']
            
        if 'stratified_checkbox' in ui_components:
            ui_components['stratified_checkbox'].value = config['data']['split']['stratified']
            
        if 'random_seed_slider' in ui_components:
            ui_components['random_seed_slider'].value = config['data']['random_seed']
            
        if 'backup_checkbox' in ui_components:
            ui_components['backup_checkbox'].value = config['data']['backup_before_split']
            
        if 'backup_dir' in ui_components:
            ui_components['backup_dir'].value = config['data']['backup_dir']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi split dataset")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")

def load_config_with_manager(config_manager: Any, module_name: str) -> Dict[str, Any]:
    """
    Load konfigurasi menggunakan config manager.
    
    Args:
        config_manager: Config manager instance
        module_name: Nama modul
        
    Returns:
        Dictionary konfigurasi
    """
    try:
        return config_manager.get_module_config(module_name) or get_default_split_config()
    except Exception as e:
        logger.error(f"❌ Error saat memuat konfigurasi: {str(e)}")
        return get_default_split_config()

def save_config_with_manager(config_manager: Any, module_name: str, config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi menggunakan config manager.
    
    Args:
        config_manager: Config manager instance
        module_name: Nama modul
        config: Konfigurasi yang akan disimpan
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        return config_manager.save_module_config(module_name, config)
    except Exception as e:
        logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False
