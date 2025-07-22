"""
File: smartcash/ui/dataset/downloader/configs/downloader_defaults.py
Deskripsi: Default configuration untuk dataset downloader module
"""

from typing import Dict, Any, List
from datetime import datetime
from .downloader_config_constants import get_default_config_structure

def get_default_downloader_config() -> Dict[str, Any]:
    """Get default downloader configuration"""
    config = get_default_config_structure()
    
    # Add timestamp information
    current_time = datetime.now().isoformat()
    config['history']['created_at'] = current_time
    config['history']['updated_at'] = current_time
    
    return config

def get_preset_workspaces() -> List[Dict[str, str]]:
    """Get list of preset workspaces for quick selection"""
    return [
        {'name': 'SmartCash', 'id': 'smartcash-wo2us'},
        {'name': 'Public Datasets', 'id': 'public'}
    ]

def get_supported_formats() -> List[Dict[str, str]]:
    """Get supported dataset formats"""
    return [
        {'id': 'yolov5pytorch', 'name': 'YOLOv5 PyTorch'},
        {'id': 'coco', 'name': 'COCO JSON'},
        {'id': 'pascalvoc', 'name': 'Pascal VOC XML'},
        {'id': 'tensorflow', 'name': 'TensorFlow TFRecord'}
    ]

def get_naming_strategies() -> List[Dict[str, str]]:
    """Get available naming strategies"""
    return [
        {'id': 'research_uuid', 'name': 'Research UUID'},
        {'id': 'original', 'name': 'Original Filenames'},
        {'id': 'sequential', 'name': 'Sequential Numbering'}
    ]

# Alias for backward compatibility
get_default_config = get_default_downloader_config

def get_roboflow_defaults() -> Dict[str, str]:
    """Get default Roboflow configuration"""
    config = get_default_downloader_config()
    return config['data']['roboflow'].copy()

def get_download_defaults() -> Dict[str, Any]:
    """Get default download settings"""
    config = get_default_downloader_config()
    return config['download'].copy()

def get_uuid_defaults() -> Dict[str, Any]:
    """Get default UUID renaming settings"""
    config = get_default_downloader_config()
    return config['uuid_renaming'].copy()

# One-liner utilities for quick access
get_default_workspace = lambda: 'smartcash-wo2us'
get_default_project = lambda: 'rupiah-emisi-2022'
get_default_version = lambda: '3'
is_uuid_enabled_by_default = lambda: True
is_validation_enabled_by_default = lambda: True