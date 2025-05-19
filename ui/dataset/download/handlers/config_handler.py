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

logger = get_logger(__name__)

def load_default_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk download dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    # Nilai default untuk download dataset
    return {
        'data': {
            'download': {
                'source': 'roboflow',
                'output_dir': 'data/downloads',
                'backup_before_download': True,
                'backup_dir': 'data/downloads_backup'
            },
            'roboflow': {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3',
                'api_key': ''
            }
        }
    }

def load_config() -> Dict[str, Any]:
    """
    Muat konfigurasi download dataset dari file YAML atau ConfigManager.
    Returns:
        Dictionary konfigurasi
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        config_manager = get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
        config = config_manager.get_module_config('dataset_download')
        if config:
            logger.info(f"💾 Konfigurasi dataset dimuat dari ConfigManager")
            return config
        else:
            logger.debug(f"ℹ️ Konfigurasi dataset tidak ditemukan di ConfigManager, menggunakan default")
    except Exception as e:
        logger.warning(f"⚠️ Error saat mengakses ConfigManager: {str(e)}")
    # Fallback ke file
    config_path = Path("configs/dataset_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"💾 Konfigurasi dataset dimuat dari {config_path}")
        return config
    logger.warning(f"⚠️ File konfigurasi tidak ditemukan, menggunakan konfigurasi default")
    return load_default_config()

def save_config(config: Dict[str, Any], logger=None) -> str:
    """
    Simpan konfigurasi download dataset ke file YAML atau ConfigManager.
    Args:
        config: Konfigurasi yang akan disimpan
        logger: Logger untuk logging
    Returns:
        Path ke file konfigurasi yang disimpan
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        config_manager = get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
        config_manager.save_module_config('dataset_download', config)
        if logger:
            logger.info(f"💾 Konfigurasi dataset berhasil disimpan menggunakan ConfigManager")
        return str(config_manager._get_module_config_path('dataset_download'))
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error saat menyimpan dengan ConfigManager: {str(e)}")
    # Fallback ke file
    config_path = Path("configs/dataset_config.yaml")
    os.makedirs(config_path.parent, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    if logger:
        logger.info(f"💾 Konfigurasi dataset berhasil disimpan ke {config_path}")
    return str(config_path)

def get_config_manager_instance():
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        return get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
    except Exception as e:
        logger.error(f"❌ Error saat mendapatkan ConfigManager instance: {str(e)}")
        return None

def save_config_with_manager(config: Dict[str, Any], ui_components: Dict[str, Any], logger=None) -> bool:
    config_manager = get_config_manager_instance()
    if config_manager:
        try:
            config_manager.register_ui_components('dataset_download', ui_components)
            return config_manager.save_module_config('dataset_download', config)
        except Exception as e:
            if logger:
                logger.warning(f"Gagal menyimpan dengan ConfigManager: {str(e)}")
    return bool(save_config(config, logger))

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
    if 'download' not in config['data']:
        config['data']['download'] = {}
    if 'roboflow' not in config['data']:
        config['data']['roboflow'] = {}
    
    # Update nilai output_dir dari input
    if 'output_dir' in ui_components:
        config['data']['download']['output_dir'] = ui_components['output_dir'].value
    
    # Update nilai source dari dropdown
    if 'source_dropdown' in ui_components:
        config['data']['download']['source'] = ui_components['source_dropdown'].value
    
    # Update nilai roboflow dari input fields
    if 'workspace' in ui_components:
        config['data']['roboflow']['workspace'] = ui_components['workspace'].value
    if 'project' in ui_components:
        config['data']['roboflow']['project'] = ui_components['project'].value
    if 'version' in ui_components:
        config['data']['roboflow']['version'] = ui_components['version'].value
    if 'api_key' in ui_components:
        config['data']['roboflow']['api_key'] = ui_components['api_key'].value
    
    # Update nilai backup dari checkbox
    if 'backup_checkbox' in ui_components:
        config['data']['download']['backup_before_download'] = ui_components['backup_checkbox'].value
    
    # Update nilai backup dir dari input
    if 'backup_dir' in ui_components:
        config['data']['download']['backup_dir'] = ui_components['backup_dir'].value
    
    return config

def update_ui_from_config(config: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Update UI dari nilai konfigurasi.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
    """
    # Jika config kosong, gunakan default
    if not config:
        config = load_default_config()
    
    # Pastikan struktur konfigurasi benar
    if 'data' not in config:
        config['data'] = {}
    if 'download' not in config['data']:
        config['data']['download'] = {}
    if 'roboflow' not in config['data']:
        config['data']['roboflow'] = {}
    
    # Update UI dari konfigurasi
    download_config = config['data'].get('download', {})
    roboflow_config = config['data'].get('roboflow', {})
    
    # Update nilai output_dir dari konfigurasi
    if 'output_dir' in ui_components and 'output_dir' in download_config:
        ui_components['output_dir'].value = download_config['output_dir']
    
    # Update nilai source dari konfigurasi
    if 'source_dropdown' in ui_components and 'source' in download_config:
        ui_components['source_dropdown'].value = download_config['source']
    
    # Update nilai roboflow dari konfigurasi
    if 'workspace' in ui_components and 'workspace' in roboflow_config:
        ui_components['workspace'].value = roboflow_config['workspace']
    if 'project' in ui_components and 'project' in roboflow_config:
        ui_components['project'].value = roboflow_config['project']
    if 'version' in ui_components and 'version' in roboflow_config:
        ui_components['version'].value = roboflow_config['version']
    if 'api_key' in ui_components and 'api_key' in roboflow_config:
        ui_components['api_key'].value = roboflow_config['api_key']
    
    # Update nilai backup dari konfigurasi
    if 'backup_checkbox' in ui_components and 'backup_before_download' in download_config:
        ui_components['backup_checkbox'].value = download_config['backup_before_download']
    
    # Update nilai backup dir dari konfigurasi
    if 'backup_dir' in ui_components and 'backup_dir' in download_config:
        ui_components['backup_dir'].value = download_config['backup_dir']

def get_dataset_manager() -> DatasetManager:
    """Get the dataset manager instance."""
    return DatasetManager()

def get_download_service() -> DownloadService:
    """Get the download service instance."""
    return DownloadService()
