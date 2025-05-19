"""
File: smartcash/ui/dataset/download/handlers/config_handler.py
Deskripsi: Handler untuk operasi konfigurasi download dataset
"""

from typing import Dict, Any, Optional
import yaml
import json
import os
from pathlib import Path
import time
from datetime import datetime
import shutil

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.dataset.manager import DatasetManager
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.common.config import ConfigManager, get_config_manager

logger = get_logger(__name__)

def get_default_download_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk download dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    return {
        'download': {
            'source': 'roboflow',
            'output_dir': 'data/downloads',
            'backup_before_download': True,
            'backup_dir': 'data/downloads_backup',
            'roboflow': {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3',
                'api_key': ''
            }
        }
    }

def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi download dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi download
    """
    logger = ui_components.get('logger', get_logger('download'))
    
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get base config
        config = config_manager.get_config()
        
        # Get download options from UI
        if 'output_dir' in ui_components:
            config['download'] = config.get('download', {})
            config['download']['output_dir'] = ui_components['output_dir'].value
            
        if 'source_dropdown' in ui_components:
            config['download']['source'] = ui_components['source_dropdown'].value
            
        if 'workspace' in ui_components:
            config['download']['roboflow'] = config.get('download', {}).get('roboflow', {})
            config['download']['roboflow']['workspace'] = ui_components['workspace'].value
            
        if 'project' in ui_components:
            config['download']['roboflow']['project'] = ui_components['project'].value
            
        if 'version' in ui_components:
            config['download']['roboflow']['version'] = ui_components['version'].value
            
        if 'api_key' in ui_components:
            config['download']['roboflow']['api_key'] = ui_components['api_key'].value
            
        if 'backup_checkbox' in ui_components:
            config['download']['backup_before_download'] = ui_components['backup_checkbox'].value
            
        if 'backup_dir' in ui_components:
            config['download']['backup_dir'] = ui_components['backup_dir'].value
            
        logger.info("✅ Konfigurasi download berhasil diupdate dari UI (tanpa update_config)")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi dari UI: {str(e)}")
        return get_default_download_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi download dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger('download'))
    
    try:
        # Get config from UI
        config = get_config_from_ui(ui_components)
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Update config in manager
        config_manager.update_config(config)
        
        logger.info("✅ Konfigurasi download berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi: {str(e)}")
        return get_default_download_config()

def get_download_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi download terbaru.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi download
    """
    logger = ui_components.get('logger', get_logger('download'))
    
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_config()
        
        # Get download config
        download_config = config.get('download', {})
        
        if not download_config:
            logger.warning("⚠️ Konfigurasi download tidak ditemukan, menggunakan default")
            download_config = get_default_download_config()['download']
            
        return download_config
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi download: {str(e)}")
        return get_default_download_config()['download']

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Dict[str, Any] = None) -> None:
    """
    Update komponen UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config_to_use: Konfigurasi yang akan digunakan
    """
    logger = ui_components.get('logger', get_logger('download'))
    
    try:
        # Get config
        if config_to_use:
            config = config_to_use
        else:
            config = get_download_config(ui_components)
        
        # Update UI components
        if 'output_dir' in ui_components and 'output_dir' in config:
            ui_components['output_dir'].value = config['output_dir']
            
        if 'source_dropdown' in ui_components and 'source' in config:
            ui_components['source_dropdown'].value = config['source']
            
        if 'workspace' in ui_components and 'roboflow' in config:
            ui_components['workspace'].value = config['roboflow'].get('workspace', '')
            
        if 'project' in ui_components and 'roboflow' in config:
            ui_components['project'].value = config['roboflow'].get('project', '')
            
        if 'version' in ui_components and 'roboflow' in config:
            ui_components['version'].value = config['roboflow'].get('version', '')
            
        if 'api_key' in ui_components and 'roboflow' in config:
            ui_components['api_key'].value = config['roboflow'].get('api_key', '')
            
        if 'backup_checkbox' in ui_components and 'backup_before_download' in config:
            ui_components['backup_checkbox'].value = config['backup_before_download']
            
        if 'backup_dir' in ui_components and 'backup_dir' in config:
            ui_components['backup_dir'].value = config['backup_dir']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")

def get_dataset_manager() -> DatasetManager:
    """Get the dataset manager instance."""
    return DatasetManager()

def get_download_service() -> DownloadService:
    """Get the download service instance."""
    return DownloadService()

def save_config_with_manager(*args, **kwargs):
    """
    Stub for save_config_with_manager. Returns True for now.
    """
    return True

def load_config(*args, **kwargs):
    """
    Stub for load_config. Returns the default download config for now.
    """
    return get_default_download_config()
