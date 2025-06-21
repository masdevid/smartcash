"""
File: smartcash/ui/dataset/downloader/handlers/defaults.py
Deskripsi: Hardcoded default configuration untuk reset operations tanpa dependency ke yaml files
"""

from typing import Dict, Any

def get_default_downloader_config() -> Dict[str, Any]:
    """
    Get hardcoded default configuration untuk downloader reset operations.
    Tidak bergantung pada yaml files untuk menghindari circular dependency.
    
    Returns:
        Dictionary berisi default configuration
    """
    return {
        'data': {
            'source': 'roboflow',
            'roboflow': {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3',
                'api_key': '',
                'output_format': 'yolov5pytorch'
            },
            'file_naming': {
                'uuid_format': True,
                'naming_strategy': 'research_uuid',
                'preserve_original': False
            }
        },
        'download': {
            'rename_files': True,
            'organize_dataset': True,
            'validate_download': True,
            'backup_existing': False,
            'retry_count': 3,
            'timeout': 30,
            'chunk_size': 8192
        },
        'uuid_renaming': {
            'enabled': True,
            'backup_before_rename': False,
            'batch_size': 1000,
            'parallel_workers': 4,
            'validate_consistency': True
        }
    }

def get_roboflow_defaults() -> Dict[str, str]:
    """Get default Roboflow configuration untuk UI reset dengan one-liner"""
    config = get_default_downloader_config()
    roboflow = config['data']['roboflow']
    return {
        'workspace': roboflow['workspace'],
        'project': roboflow['project'], 
        'version': roboflow['version'],
        'api_key': roboflow['api_key']
    }

def get_download_defaults() -> Dict[str, Any]:
    """Get default download options untuk UI reset dengan one-liner"""
    download = get_default_downloader_config()['download']
    return {
        'validate_download': download['validate_download'],
        'backup_existing': download['backup_existing'],
        'rename_files': download['rename_files'],
        'organize_dataset': download['organize_dataset']
    }

def get_uuid_defaults() -> Dict[str, Any]:
    """Get default UUID renaming settings untuk reset dengan one-liner"""
    uuid_config = get_default_downloader_config()['uuid_renaming']
    return {
        'enabled': uuid_config['enabled'],
        'backup_before_rename': uuid_config['backup_before_rename'],
        'validate_consistency': uuid_config['validate_consistency']
    }

# One-liner utilities untuk quick access
get_default_workspace = lambda: 'smartcash-wo2us'
get_default_project = lambda: 'rupiah-emisi-2022'
get_default_version = lambda: '3'
is_uuid_enabled_by_default = lambda: True
is_validation_enabled_by_default = lambda: True