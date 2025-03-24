"""
File: smartcash/ui/dataset/handlers/config_loader.py
Deskripsi: Utilitas untuk load konfigurasi dataset dan Roboflow dari berbagai sumber
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def load_dataset_config(config: Optional[Dict[str, Any]] = None, logger = None) -> Dict[str, Any]:
    """
    Load konfigurasi dataset dari berbagai sumber dengan prioritas.
    
    Args:
        config: Konfigurasi yang sudah ada (opsional)
        logger: Logger kustom (opsional)
        
    Returns:
        Konfigurasi dataset yang telah diload
    """
    # Konfigurasi default
    default_config = {
        'roboflow': {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'api_key': ''
        },
        'dir': 'data',
        'split_ratios': {
            'train': 0.7,
            'valid': 0.15,
            'test': 0.15
        }
    }
    
    # Gunakan konfigurasi yang sudah ada jika disediakan
    dataset_config = (config or {}).get('data', {})
    
    # Coba load dari file dataset_config.yaml
    file_config = _load_from_dataset_config()
    
    # Gabungkan dengan prioritas:
    # 1. Config yang sudah ada
    # 2. File dataset_config.yaml
    # 3. Default config
    merged_config = {**default_config, **(file_config or {}), **dataset_config}
    
    # Override dengan environment variables jika ada
    roboflow_config = merged_config.get('roboflow', {})
    api_key = os.environ.get('ROBOFLOW_API_KEY', roboflow_config.get('api_key', ''))
    roboflow_config['api_key'] = api_key
    merged_config['roboflow'] = roboflow_config
    
    # Log jika tersedia
    if logger:
        if api_key:
            logger.info("✅ API key Roboflow loaded dari environment variable")
        if file_config:
            logger.info("✅ Konfigurasi dataset berhasil diload dari file")
    
    return merged_config

def _load_from_dataset_config() -> Optional[Dict[str, Any]]:
    """
    Load konfigurasi dari file dataset_config.yaml.
    
    Returns:
        Konfigurasi dari file atau None jika gagal
    """
    try:
        # Coba import config_manager
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        
        # Cek beberapa kemungkinan path file
        config_files = ['dataset_config.yaml', 'configs/dataset_config.yaml']
        
        for file_path in config_files:
            try:
                config = config_manager.load_config(file_path)
                if config:
                    return config.get('data', {})
            except Exception:
                continue
    except ImportError:
        # Fallback: baca file langsung
        try:
            import yaml
            config_files = ['dataset_config.yaml', 'configs/dataset_config.yaml']
            
            for file_path in config_files:
                path = Path(file_path)
                if path.exists():
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                        if config:
                            return config.get('data', {})
        except Exception:
            pass
    
    return None

def get_roboflow_credentials() -> Tuple[str, str, str, str]:
    """
    Dapatkan kredensial Roboflow dari berbagai sumber.
    
    Returns:
        Tuple (workspace, project, version, api_key)
    """
    # Load konfigurasi
    data_config = load_dataset_config()
    roboflow_config = data_config.get('roboflow', {})
    
    # Dapatkan nilai-nilai
    workspace = roboflow_config.get('workspace', '')
    project = roboflow_config.get('project', '')
    version = roboflow_config.get('version', '')
    api_key = roboflow_config.get('api_key', '')
    
    return workspace, project, version, api_key