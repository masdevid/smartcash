"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler untuk konfigurasi split dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
import os
from pathlib import Path

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_split_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi split dataset
    """
    try:
        # Get config manager
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Get config
        config = config_manager.get_module_config('dataset')
        
        # Ensure config structure
        if not config:
            config = get_default_split_config()
        elif 'data' not in config:
            config['data'] = {}
        elif 'split' not in config['data']:
            config['data']['split'] = get_default_split_config()['data']['split']
            
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat get split config: {str(e)}")
        return get_default_split_config()

def get_default_split_config() -> Dict[str, Any]:
    """
    Get konfigurasi default split dataset.
    
    Returns:
        Dictionary konfigurasi default split dataset
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
            'backup_dir': 'backup'
        }
    }

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get current config
        config = get_split_config(ui_components)
        
        # Update split config
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
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config_manager.save_module_config('dataset', config)
        
        logger.info("✅ Konfigurasi berhasil diupdate dari UI")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update config dari UI: {str(e)}")
        return get_split_config(ui_components)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Ensure config structure
        if not config:
            config = get_default_split_config()
        elif 'data' not in config:
            config['data'] = {}
        elif 'split' not in config['data']:
            config['data']['split'] = get_default_split_config()['data']['split']
            
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
            
        # Update total label
        if 'total_label' in ui_components:
            total = round(config['data']['split']['train'] + config['data']['split']['val'] + config['data']['split']['test'], 2)
            color = 'green' if total == 1.0 else 'red'
            ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error saat update UI dari config: {str(e)}")
        return ui_components
