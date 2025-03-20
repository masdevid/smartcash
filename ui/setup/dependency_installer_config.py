"""
File: smartcash/ui/setup/dependency_installer_config.py
Deskripsi: Konfigurasi dependencies untuk SmartCash
"""

import yaml
from typing import Dict, Any

def get_dependency_config(config_path: str = "configs/colab_config.yaml") -> Dict[str, Any]:
    """Ambil konfigurasi dependencies dari file."""
    default_config = {'dependencies': {'core_packages': ['yolov5_requirements', 'smartcash_utils', 'notebook_tools'], 'ml_packages': ['pytorch', 'opencv', 'albumentations'], 'viz_packages': ['matplotlib', 'pandas', 'seaborn']}, 'install_options': {'force_reinstall': False, 'user_install': False, 'upgrade': False}}
    try: return default_config.update(yaml.safe_load(open(config_path, 'r')).get('dependencies', {})) or default_config
    except FileNotFoundError: return default_config
    except Exception as e: return print(f"⚠️ Error membaca konfigurasi: {e}") or default_config

def validate_dependency_config(config: Dict[str, Any]) -> bool:
    """Validasi konfigurasi dependencies."""
    required_keys = ['dependencies', 'install_options']
    return all(key in config for key in required_keys)