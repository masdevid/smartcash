"""
File: smartcash/ui/dataset/download/handlers/config_handler.py
Deskripsi: Handler untuk operasi konfigurasi download dataset
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.dataset.manager import DatasetManager
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.common.config import get_config_manager

logger = get_logger(__name__)

def map_config_to_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map konfigurasi dari struktur config ke struktur form UI.
    
    Args:
        config: Konfigurasi dari file YAML
        
    Returns:
        Dictionary dengan struktur form UI
    """
    form_config = {
        'download': {
            'source': config.get('data', {}).get('source', 'roboflow'),
            'output_dir': 'data/downloads',  # Default value
            'backup_before_download': config.get('dataset', {}).get('backup', {}).get('enabled', True),
            'backup_dir': config.get('dataset', {}).get('backup', {}).get('dir', 'data/backup/dataset'),
            'roboflow': {
                'workspace': config.get('data', {}).get('roboflow', {}).get('workspace', ''),
                'project': config.get('data', {}).get('roboflow', {}).get('project', ''),
                'version': config.get('data', {}).get('roboflow', {}).get('version', ''),
                'api_key': config.get('data', {}).get('roboflow', {}).get('api_key', '')
            }
        }
    }
    return form_config

def map_form_to_config(form_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map konfigurasi dari struktur form UI ke struktur config.
    
    Args:
        form_config: Konfigurasi dari form UI
        
    Returns:
        Dictionary dengan struktur config YAML
    """
    download_config = form_config.get('download', {})
    roboflow_config = download_config.get('roboflow', {})
    
    config = {
        'data': {
            'source': download_config.get('source', 'roboflow'),
            'roboflow': {
                'workspace': roboflow_config.get('workspace', ''),
                'project': roboflow_config.get('project', ''),
                'version': roboflow_config.get('version', ''),
                'api_key': roboflow_config.get('api_key', '')
            }
        },
        'dataset': {
            'backup': {
                'enabled': download_config.get('backup_before_download', True),
                'dir': download_config.get('backup_dir', 'data/backup/dataset')
            }
        }
    }
    return config

def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi download dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi download
    """
    logger = ui_components.get('logger', get_logger())
    
    try:
        # Get config manager (dengan fallback otomatis)
        config_manager = get_config_manager()
        
        # Get base config
        config = config_manager.config
        
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
            
        logger.info("✅ Konfigurasi download berhasil diupdate dari UI")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi dari UI: {str(e)}")
        raise

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi download dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger())
    
    try:
        # Get config from UI
        form_config = get_config_from_ui(ui_components)
        
        # Convert form config to YAML config structure
        config = map_form_to_config(form_config)
        
        # Get config manager (dengan fallback otomatis)
        config_manager = get_config_manager()
        
        # Update config in manager
        config_manager.update_config(config)
        
        logger.info("✅ Konfigurasi download berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi: {str(e)}")
        raise

def get_download_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi download terbaru.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi download
    """
    logger = ui_components.get('logger', get_logger())
    
    try:
        # Get config manager (dengan fallback otomatis)
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.config
        logger.info(f"Loaded config: {config}")
        
        # Convert YAML config to form config structure
        form_config = map_config_to_form(config)
        logger.info(f"Mapped form config: {form_config}")
        
        # Get download config
        download_config = form_config.get('download', {})
        if not download_config:
            logger.warning("⚠️ Konfigurasi download tidak ditemukan")
            raise ValueError("Konfigurasi download tidak ditemukan")
            
        return download_config
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi download: {str(e)}")
        raise

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Dict[str, Any] = None) -> None:
    """
    Update komponen UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config_to_use: Konfigurasi yang akan digunakan
    """
    logger = ui_components.get('logger', get_logger())
    
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
            ui_components['workspace'].value = config['roboflow']['workspace']
            
        if 'project' in ui_components and 'roboflow' in config:
            ui_components['project'].value = config['roboflow']['project']
            
        if 'version' in ui_components and 'roboflow' in config:
            ui_components['version'].value = config['roboflow']['version']
            
        if 'api_key' in ui_components and 'roboflow' in config:
            ui_components['api_key'].value = config['roboflow']['api_key']
            
        if 'backup_checkbox' in ui_components and 'backup_before_download' in config:
            ui_components['backup_checkbox'].value = config['backup_before_download']
            
        if 'backup_dir' in ui_components and 'backup_dir' in config:
            ui_components['backup_dir'].value = config['backup_dir']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi")
        
    except Exception as e:
        logger.error(f"❌ Error saat update UI dari konfigurasi: {str(e)}")
        raise

def get_dataset_manager() -> DatasetManager:
    """
    Dapatkan instance DatasetManager.
    
    Returns:
        Instance DatasetManager
    """
    return DatasetManager()

def get_download_service() -> DownloadService:
    """
    Dapatkan instance DownloadService.
    
    Returns:
        Instance DownloadService
    """
    return DownloadService()
