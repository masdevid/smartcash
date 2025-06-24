"""
File: smartcash/ui/setup/dependency/handlers/defaults.py
Deskripsi: Default configuration untuk dependency installer dengan struktur yang sama dengan YAML
"""

from typing import Dict, Any
import copy

# Default configuration yang selaras dengan dependency_config.yaml
DEFAULT_CONFIG = {
    'module_name': 'dependency',
    'version': '1.0.0',
    'created_by': 'SmartCash',
    'description': 'Dependency installer configuration untuk SmartCash project',
    
    # Package selections
    'selected_packages': [],
    'custom_packages': '',
    'auto_analyze': True,
    
    # Installation settings
    'installation': {
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'timeout': 300,
        'max_retries': 2,
        'retry_delay': 1.0
    },
    
    # Analysis settings
    'analysis': {
        'check_compatibility': True,
        'include_dev_deps': False,
        'batch_size': 10,
        'detailed_info': True
    },
    
    # UI settings
    'ui_settings': {
        'auto_analyze_on_render': True,
        'show_progress': True,
        'log_level': 'info',
        'compact_view': False
    },
    
    # Advanced settings
    'advanced': {
        'pip_extra_args': [],
        'environment_variables': {},
        'pre_install_commands': [],
        'post_install_commands': []
    }
}

def get_default_dependency_config() -> Dict[str, Any]:
    """Get complete default config untuk dependency installer"""
    return copy.deepcopy(DEFAULT_CONFIG)

def get_minimal_config() -> Dict[str, Any]:
    """Get minimal config untuk basic functionality"""
    return {
        'module_name': DEFAULT_CONFIG['module_name'],
        'selected_packages': DEFAULT_CONFIG['selected_packages'],
        'installation': DEFAULT_CONFIG['installation'],
        'analysis': DEFAULT_CONFIG['analysis']
    }

def get_environment_specific_config() -> Dict[str, Any]:
    """Get config yang disesuaikan dengan environment"""
    config = get_default_dependency_config()
    
    # Sesuaikan dengan environment Google Colab
    config['installation']['parallel_workers'] = 2  # Lebih konservatif di Colab
    config['installation']['timeout'] = 600  # Timeout lebih lama
    config['ui_settings']['compact_view'] = True  # Compact view di Colab
    
    return config